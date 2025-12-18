# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 16:52:43 2022

@author: xiong
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from segmentation_models_pytorch.encoders import get_encoder
# 修正 Decoder 导入路径
from segmentation_models_pytorch.decoders.fpn.decoder import FPNDecoder
# 导入层次映射关系
from hierarchy_dict import map_3_to_2, map_2_to_1

# 尝试导入CLIP
try:
    import clip
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    print("Warning: CLIP not available. Text-guided features will be disabled.")

# 层次一致性损失函数
class HierarchyConsistencyLoss(nn.Module):
    """
    确保三级标签预测与二级标签预测之间保持正确的层次关系（林草数据集）
    
    参数:
        map_3_to_2: 三级类别到二级类别的映射字典（林草数据集）
        weight: 损失权重
    """
    def __init__(self, map_3_to_2=map_3_to_2, weight=0.5):
        super().__init__()
        self.weight = weight
        # 创建映射张量，用于将三级预测转换为二级预测
        max_key = max(map_3_to_2.keys())
        self.mapping = torch.zeros(max_key + 1, dtype=torch.long)
        for k, v in map_3_to_2.items():
            self.mapping[k] = v
        
    def forward(self, level2_pred, level3_pred):
        """
        计算层次一致性损失
        
        参数:
            level2_pred: [B, C2, H, W] 二级类别预测的 logits
            level3_pred: [B, C3, H, W] 三级类别预测的 logits
            
        返回:
            一致性损失值
        """
        # 获取三级预测的类别索引
        level3_indices = level3_pred.argmax(dim=1)  # [B, H, W]
        
        # 将三级索引映射到二级索引
        mapping = self.mapping.to(level3_indices.device)
        mapped_level2 = mapping[level3_indices]  # [B, H, W]
        
        # 获取二级预测的类别索引
        level2_indices = level2_pred.argmax(dim=1)  # [B, H, W]
        
        # 计算不一致的像素比例作为损失
        inconsistency = (mapped_level2 != level2_indices).float().mean()
        return self.weight * inconsistency

class AdaptiveSemanticGNN(nn.Module):
    """
    自适应语义图神经网络 - 高效实现版本
    创新点：
    1. 动态学习类别间的语义相似性关系，而非仅基于层次结构
    2. 结合空间上下文信息进行图推理
    3. 多尺度特征融合的图卷积
    
    优化：
    1. 使用向量化操作替代循环
    2. 批量计算邻接矩阵
    3. 减少动态图构建
    """
    def __init__(self, num_classes=16, hidden_dim=128):
        super().__init__()
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        
        # 可学习的语义嵌入
        self.semantic_embeddings = nn.Parameter(torch.randn(num_classes, hidden_dim))
        
        # 简化的邻接矩阵生成器 - 使用矩阵乘法替代循环
        self.adj_fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.adj_fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        
        # 图卷积层 - 使用标准的线性层实现
        self.gnn_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(2)  # 减少到2层
        ])
        
        # 特征投影层
        self.feat_proj = nn.Conv2d(num_classes, hidden_dim, 1)  # 使用1x1卷积
        self.output_proj = nn.Conv2d(hidden_dim, num_classes, 1)
        
        # 门控机制
        self.gate = nn.Parameter(torch.tensor(0.1))  # 初始化为较小值
        
    def generate_adjacency_matrix_efficient(self):
        """高效生成邻接矩阵 - 使用向量化操作"""
        # 将语义嵌入投影到低维空间
        embed_proj1 = torch.relu(self.adj_fc1(self.semantic_embeddings))  # [C, D/2]
        embed_proj2 = torch.relu(self.adj_fc2(self.semantic_embeddings))  # [C, D/2]
        
        # 计算相似性矩阵 - 使用点积
        adj_matrix = torch.matmul(embed_proj1, embed_proj2.T)  # [C, C]
        
        # 归一化并添加自环
        adj_matrix = torch.sigmoid(adj_matrix)
        adj_matrix = adj_matrix + torch.eye(self.num_classes, device=adj_matrix.device)
        
        # 行归一化
        adj_matrix = adj_matrix / adj_matrix.sum(dim=1, keepdim=True)
        
        return adj_matrix
    
    def forward(self, class_logits):
        """
        Args:
            class_logits: [B, C, H, W] 类别预测logits
        Returns:
            refined_logits: [B, C, H, W] 增强后的预测logits
        """
        B, C, H, W = class_logits.shape
        
        # 生成邻接矩阵（所有批次共享）
        adj_matrix = self.generate_adjacency_matrix_efficient()  # [C, C]
        
        # 将logits投影到隐藏空间
        hidden_feats = self.feat_proj(class_logits)  # [B, D, H, W]
        
        # 转换为 [B, H*W, D] 格式进行图卷积
        hidden_feats = hidden_feats.view(B, self.hidden_dim, -1).transpose(1, 2)  # [B, H*W, D]
        
        # 获取类别概率分布用于加权
        class_probs = F.softmax(class_logits, dim=1)  # [B, C, H, W]
        class_probs = class_probs.view(B, C, -1).transpose(1, 2)  # [B, H*W, C]
        
        # 图卷积操作
        for i, gnn_layer in enumerate(self.gnn_layers):
            # 计算加权特征聚合
            # 使用类别概率作为权重，通过邻接矩阵传播信息
            weighted_probs = torch.matmul(class_probs, adj_matrix)  # [B, H*W, C]
            
            # 将聚合的概率转换为特征向量（使用语义嵌入）
            # 扩展语义嵌入以匹配批次大小
            semantic_embed_batch = self.semantic_embeddings.unsqueeze(0).expand(B, -1, -1)  # [B, C, D]
            
            # 使用加权概率聚合语义特征
            aggregated_feats = torch.bmm(weighted_probs, semantic_embed_batch)  # [B, H*W, D]
            
            # 应用GNN层并添加残差连接
            hidden_feats = gnn_layer(hidden_feats) + aggregated_feats
            
            # 非线性激活（最后一层除外）
            if i < len(self.gnn_layers) - 1:
                hidden_feats = F.relu(hidden_feats)
        
        # 转换回 [B, D, H, W] 格式
        hidden_feats = hidden_feats.transpose(1, 2).view(B, self.hidden_dim, H, W)
        
        # 输出投影
        refined_logits = self.output_proj(hidden_feats)  # [B, C, H, W]
        
        # 门控残差连接
        return self.gate * refined_logits + class_logits

class CLIPTextGuidedAttention(nn.Module):
    """
    CLIP文本引导的交叉注意力模块
    
    创新点：
    1. 使用样本特定的CLIP文本描述引导视觉特征提取
    2. 通过交叉注意力机制让文本特征增强视觉特征
    3. 支持多尺度特征融合
    4. 轻量级设计，显存友好
    
    参数:
        visual_dim: 视觉特征的通道数
        text_dim: CLIP文本特征维度（ViT-B/32: 512）
        hidden_dim: 隐藏层维度
        num_heads: 多头注意力的头数
        dropout: Dropout比例
    """
    def __init__(self, visual_dim, text_dim=512, hidden_dim=256, num_heads=4, dropout=0.1):
        super().__init__()
        self.visual_dim = visual_dim
        self.text_dim = text_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # 视觉特征投影
        self.visual_proj = nn.Sequential(
            nn.Conv2d(visual_dim, hidden_dim, 1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        
        # 文本特征投影
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True)
        )
        
        # 多头交叉注意力（兼容旧版本PyTorch，不使用batch_first）
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # 输出投影
        self.output_proj = nn.Sequential(
            nn.Conv2d(hidden_dim, visual_dim, 1),
            nn.BatchNorm2d(visual_dim)
        )
        
        # 门控机制：动态调整文本引导的强度
        self.gate = nn.Sequential(
            nn.Conv2d(visual_dim * 2, visual_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, visual_features, text_features):
        """
        前向传播（兼容旧版本PyTorch）
        
        参数:
            visual_features: [B, C, H, W] 视觉特征图
            text_features: [B, text_dim] CLIP文本特征
        
        返回:
            enhanced_features: [B, C, H, W] 文本引导增强后的视觉特征
        """
        B, C, H, W = visual_features.shape
        
        # 1. 投影视觉特征
        visual_proj = self.visual_proj(visual_features)  # [B, hidden_dim, H, W]
        visual_flat = visual_proj.flatten(2).transpose(1, 2)  # [B, H*W, hidden_dim]
        
        # 2. 投影文本特征
        text_proj = self.text_proj(text_features)  # [B, hidden_dim]
        text_expanded = text_proj.unsqueeze(0)  # [1, B, hidden_dim] 注意：这里改为第0维
        
        # 3. 转换视觉特征维度以适配MultiheadAttention（batch_first=False的默认格式）
        visual_flat = visual_flat.transpose(0, 1)  # [H*W, B, hidden_dim]
        
        # 4. 交叉注意力：文本作为query，视觉作为key和value
        # 这样可以让文本特征去"查询"视觉特征中相关的区域
        attended_features, _ = self.cross_attention(
            query=text_expanded,  # [1, B, hidden_dim]
            key=visual_flat,      # [H*W, B, hidden_dim]
            value=visual_flat     # [H*W, B, hidden_dim]
        )  # [1, B, hidden_dim]
        
        # 5. 转换回batch_first格式并广播到所有空间位置
        attended_features = attended_features.transpose(0, 1)  # [B, 1, hidden_dim]
        attended_features = attended_features.expand(-1, H*W, -1)  # [B, H*W, hidden_dim]
        attended_features = attended_features.transpose(1, 2).view(B, self.hidden_dim, H, W)
        
        # 6. 输出投影
        guided_features = self.output_proj(attended_features)  # [B, C, H, W]
        
        # 7. 门控融合：动态调整原始特征和引导特征的比例
        gate_input = torch.cat([visual_features, guided_features], dim=1)
        gate_weights = self.gate(gate_input)  # [B, C, H, W]
        
        # 融合输出
        enhanced_features = gate_weights * guided_features + (1 - gate_weights) * visual_features
        
        return enhanced_features

class CLIPTextEncoder(nn.Module):
    """
    CLIP文本编码器包装类
    
    功能：
    1. 封装CLIP模型的文本编码功能
    2. 支持批量文本编码
    3. 处理文本为None的情况（使用默认描述）
    4. 缓存文本特征以提高效率
    """
    def __init__(self, clip_model_name='ViT-B/32', device='cuda'):
        super().__init__()
        self.device = device
        
        if not CLIP_AVAILABLE:
            print("Warning: CLIP not available, text encoding will be disabled")
            self.clip_model = None
            return
        
        # 加载CLIP模型
        try:
            self.clip_model, _ = clip.load(clip_model_name, device=device)
            self.clip_model.eval()
            for param in self.clip_model.parameters():
                param.requires_grad = False
            print(f"CLIP文本编码器初始化成功: {clip_model_name}")
        except Exception as e:
            print(f"CLIP模型加载失败: {e}")
            self.clip_model = None
        
        # 默认文本描述（当样本没有文本时使用）
        self.default_text = "An aerial image of land cover with various vegetation types"
        
    def encode_texts(self, text_list):
        """
        编码文本列表
        
        参数:
            text_list: 字符串列表，每个元素是一个样本的文本描述
        
        返回:
            text_features: [B, 512] CLIP文本特征
        """
        if self.clip_model is None:
            # 如果CLIP不可用，返回零特征
            return torch.zeros(len(text_list), 512, device=self.device)
        
        # 处理None文本（替换为默认描述）
        processed_texts = [
            text if text is not None else self.default_text 
            for text in text_list
        ]
        
        # 编码文本
        with torch.no_grad():
            text_tokens = clip.tokenize(processed_texts, truncate=True).to(self.device)
            text_features = self.clip_model.encode_text(text_tokens)
            text_features = text_features.float()
            # L2归一化
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        return text_features

class CLIPGuidedSemanticGNN(nn.Module):
    """
    CLIP引导的语义图神经网络（林草数据集适配）
    
    创新点：
    1. 利用CLIP预训练模型提供的语义嵌入作为先验知识
    2. 通过知识蒸馏让可学习嵌入向CLIP嵌入对齐
    3. 结合视觉-语言跨模态信息增强类别关系建模
    4. 动态融合CLIP先验和自适应学习的语义关系
    """
    def __init__(self, num_classes=9, hidden_dim=128, 
                 clip_model_name='ViT-B/32', 
                 temperature=0.1,
                 distill_weight=0.3,
                 dataset_type='forest'):
        super().__init__()
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.temperature = temperature
        self.distill_weight = distill_weight
        self.dataset_type = dataset_type
        
        # 导入CLIP
        try:
            import clip
            self.clip_available = True
        except ImportError:
            print("Warning: CLIP not installed. Install with: pip install git+https://github.com/openai/CLIP.git")
            self.clip_available = False
        
        # 设备信息 - 从外部传入或自动检测
        # 优先使用外部传入的设备，如果没有则自动检测
        if hasattr(self, '_device_override'):
            self.device = self._device_override
        else:
            import os
            if torch.cuda.is_available():
                gpu_id = os.environ.get('CUDA_VISIBLE_DEVICES', '0')
                if ',' in gpu_id:
                    # 如果设置了多个GPU，使用第一个
                    gpu_id = gpu_id.split(',')[0]
                # 当使用CUDA_VISIBLE_DEVICES=6时，实际上是cuda:0（因为只看到一个GPU）
                self.device = torch.device('cuda:0')
            else:
                self.device = torch.device('cpu')
        
        # 初始化CLIP模型（如果可用）
        if self.clip_available:
            try:
                # 确保CLIP模型使用与主模型相同的设备
                self.clip_model, _ = clip.load(clip_model_name, device=self.device)
                self.clip_model.eval()
                for param in self.clip_model.parameters():
                    param.requires_grad = False
                
                # CLIP特征维度（ViT-B/32: 512）
                self.clip_dim = 512
                
                # 投影层：将CLIP嵌入投影到隐藏维度
                self.clip_proj = nn.Linear(self.clip_dim, hidden_dim)
                
                print(f"CLIP模型加载成功，使用设备: {self.device}")
            except Exception as e:
                print(f"CLIP模型加载失败: {e}")
                self.clip_available = False
        
        # 可学习的语义嵌入（与原始版本相同）
        self.semantic_embeddings = nn.Parameter(torch.randn(num_classes, hidden_dim))
        
        # 融合CLIP嵌入和可学习嵌入的门控机制
        self.fusion_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # 邻接矩阵生成器（考虑CLIP语义相似性）
        self.adj_fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.adj_fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        
        # 图卷积层
        self.gnn_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(2)
        ])
        
        # 特征投影层
        self.feat_proj = nn.Conv2d(num_classes, hidden_dim, 1)
        self.output_proj = nn.Conv2d(hidden_dim, num_classes, 1)
        
        # 门控残差
        self.gate = nn.Parameter(torch.tensor(0.1))
        
        # 预定义的类别文本描述（需要根据具体数据集调整）
        self.class_descriptions = self._get_class_descriptions()
        
        # 预计算CLIP嵌入
        if self.clip_available:
            try:
                clip_embeddings = self._compute_clip_embeddings()
                if clip_embeddings is not None:
                    # 确保嵌入在CPU上注册，稍后会随模型一起移动到正确设备
                    self.register_buffer('clip_embeddings', clip_embeddings.cpu())
                else:
                    self.clip_embeddings = None
            except Exception as e:
                print(f"CLIP嵌入初始化失败: {e}")
                self.clip_embeddings = None
                self.clip_available = False
        else:
            self.clip_embeddings = None
    
    def _get_class_descriptions(self):
        """
        获取类别的文本描述
        根据实际的num_classes和数据集类型动态生成描述
        """
        if self.dataset_type.lower() == 'forest':
            # 林草数据集类别描述（9类，与hierarchy_dict.py一致）
            full_descriptions = [
                "background or unknown area",                    # 0: 背景
                "Broadleaf Forest - Tree Forest Type",          # 1: 阔叶林 (→二级类别1: 乔木林)
                "Coniferous Forest - Tree Forest Type",         # 2: 针叶林 (→二级类别1: 乔木林) 
                "Shrubland - Shrub Vegetation Type",            # 3: 灌丛 (→二级类别2: 灌木林)
                "Artificial Forest - Planted Forest Type",      # 4: 人工林 (→二级类别3: 人工林地)
                "Natural Grassland - Natural Grass Type",       # 5: 天然草地 (→二级类别4: 天然草地)
                "Other Artificial Grassland - Artificial Grass Type",   # 6: 其他人工草地 (→二级类别5: 人工草地)
                "Landscaped Grassland - Artificial Grass Type",         # 7: 绿化草地 (→二级类别5: 人工草地)
                "Slope Protection Vegetation - Artificial Grass Type"   # 8: 护坡植草 (→二级类别5: 人工草地)
            ]
        else:
            # 通用描述
            full_descriptions = [f"land cover class {i}" for i in range(max(9, self.num_classes))]
        
        # 根据实际类别数量调整描述
        if self.num_classes <= len(full_descriptions):
            descriptions = full_descriptions[:self.num_classes]
        else:
            # 如果需要更多类别，扩展通用描述
            descriptions = full_descriptions[:]
            for i in range(len(full_descriptions), self.num_classes):
                descriptions.append(f"land cover class {i}")
        
        print(f"为{self.dataset_type}数据集的{self.num_classes}个类别生成描述: {len(descriptions)}个")
        return descriptions
    
    def _compute_clip_embeddings(self):
        """
        使用CLIP文本编码器计算类别的语义嵌入
        """
        if not self.clip_available:
            return None
        
        try:
            import clip
            
            # 构建更丰富的文本提示
            text_prompts = []
            for desc in self.class_descriptions:
                # 使用多个模板增强语义表示
                templates = [
                    f"a satellite image of {desc}",
                    f"an aerial view of {desc}",
                    f"remote sensing image showing {desc}",
                    f"{desc} in satellite imagery"
                ]
                text_prompts.extend(templates)
            
            print(f"正在计算{len(text_prompts)}个文本提示的CLIP嵌入...")
            
            # 编码文本
            with torch.no_grad():
                text_tokens = clip.tokenize(text_prompts).to(self.device)
                text_features = self.clip_model.encode_text(text_tokens)
                text_features = text_features.float()
                
                # 对每个类别的多个模板取平均
                num_templates = 4
                text_features = text_features.view(self.num_classes, num_templates, -1).mean(dim=1)
                
                # L2归一化
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            print(f"CLIP嵌入计算完成，形状: {text_features.shape}")
            return text_features
            
        except Exception as e:
            print(f"计算CLIP嵌入时出错: {e}")
            print("将禁用CLIP功能")
            self.clip_available = False
            return None
    
    def get_fused_embeddings(self):
        """
        融合CLIP嵌入和可学习嵌入
        """
        if self.clip_available and self.clip_embeddings is not None:
            # 确保CLIP嵌入在正确的设备上
            device = self.semantic_embeddings.device
            clip_embeddings_on_device = self.clip_embeddings.to(device)
            
            # 投影CLIP嵌入到隐藏维度
            clip_proj = self.clip_proj(clip_embeddings_on_device)
            
            # 计算融合权重
            concat_embed = torch.cat([self.semantic_embeddings, clip_proj], dim=-1)
            fusion_weights = self.fusion_gate(concat_embed)  # [num_classes, 1]
            
            # 加权融合
            fused_embeddings = fusion_weights * clip_proj + (1 - fusion_weights) * self.semantic_embeddings
        else:
            # 如果CLIP不可用，只使用可学习嵌入
            fused_embeddings = self.semantic_embeddings
        
        return fused_embeddings
    
    def generate_clip_guided_adjacency(self):
        """
        生成CLIP引导的邻接矩阵
        """
        # 获取融合后的嵌入
        fused_embeddings = self.get_fused_embeddings()
        
        # 投影到低维空间
        embed_proj1 = torch.relu(self.adj_fc1(fused_embeddings))  # [C, D/2]
        embed_proj2 = torch.relu(self.adj_fc2(fused_embeddings))  # [C, D/2]
        
        # 计算相似性矩阵
        adj_matrix = torch.matmul(embed_proj1, embed_proj2.T)  # [C, C]
        
        # 如果有CLIP嵌入，额外考虑CLIP空间的语义相似性
        if self.clip_available and self.clip_embeddings is not None:
            # 计算CLIP嵌入的余弦相似度
            clip_sim = torch.matmul(self.clip_embeddings, self.clip_embeddings.T)
            # 温度缩放
            clip_sim = clip_sim / self.temperature
            # 融合两种相似性
            adj_matrix = 0.7 * torch.sigmoid(adj_matrix) + 0.3 * torch.sigmoid(clip_sim)
        else:
            adj_matrix = torch.sigmoid(adj_matrix)
        
        # 添加自环
        adj_matrix = adj_matrix + torch.eye(self.num_classes, device=adj_matrix.device)
        
        # 行归一化
        adj_matrix = adj_matrix / adj_matrix.sum(dim=1, keepdim=True)
        
        return adj_matrix
    
    def compute_distillation_loss(self):
        """
        计算知识蒸馏损失，让可学习嵌入向CLIP嵌入对齐
        """
        if not self.clip_available or self.clip_embeddings is None:
            return 0.0
        
        # 投影CLIP嵌入
        clip_proj = self.clip_proj(self.clip_embeddings)
        
        # 计算余弦相似度损失
        cos_sim = F.cosine_similarity(self.semantic_embeddings, clip_proj, dim=-1)
        distill_loss = (1 - cos_sim).mean()
        
        return self.distill_weight * distill_loss
    
    def forward(self, class_logits):
        """
        前向传播
        
        Args:
            class_logits: [B, C, H, W] 类别预测logits
            
        Returns:
            refined_logits: [B, C, H, W] 增强后的预测logits
        """
        B, C, H, W = class_logits.shape
        
        # 生成CLIP引导的邻接矩阵
        adj_matrix = self.generate_clip_guided_adjacency()  # [C, C]
        
        # 获取融合后的语义嵌入
        fused_embeddings = self.get_fused_embeddings()  # [C, D]
        
        # 将logits投影到隐藏空间
        hidden_feats = self.feat_proj(class_logits)  # [B, D, H, W]
        
        # 转换格式进行图卷积
        hidden_feats = hidden_feats.view(B, self.hidden_dim, -1).transpose(1, 2)  # [B, H*W, D]
        
        # 获取类别概率分布
        class_probs = F.softmax(class_logits, dim=1)  # [B, C, H, W]
        class_probs = class_probs.view(B, C, -1).transpose(1, 2)  # [B, H*W, C]
        
        # 图卷积操作
        for i, gnn_layer in enumerate(self.gnn_layers):
            # 通过邻接矩阵传播概率
            weighted_probs = torch.matmul(class_probs, adj_matrix)  # [B, H*W, C]
            
            # 扩展语义嵌入
            semantic_embed_batch = fused_embeddings.unsqueeze(0).expand(B, -1, -1)  # [B, C, D]
            
            # 聚合语义特征
            aggregated_feats = torch.bmm(weighted_probs, semantic_embed_batch)  # [B, H*W, D]
            
            # 应用GNN层
            hidden_feats = gnn_layer(hidden_feats) + aggregated_feats
            
            if i < len(self.gnn_layers) - 1:
                hidden_feats = F.relu(hidden_feats)
        
        # 转换回原始格式
        hidden_feats = hidden_feats.transpose(1, 2).view(B, self.hidden_dim, H, W)
        
        # 输出投影
        refined_logits = self.output_proj(hidden_feats)  # [B, C, H, W]
        
        # 门控残差连接
        return self.gate * refined_logits + class_logits

class EfficientSelfAttention(nn.Module):
    """
    内存高效的自注意力模块，通过空间降采样减少内存占用
    
    参数:
        in_channels: 输入特征通道数
        reduction_ratio: 空间降采样比例，默认为8
    """
    def __init__(self, in_channels, reduction_ratio=8):
        super().__init__()
        self.reduction_ratio = reduction_ratio
        # 确保通道数至少为8，避免除以8后为0
        mid_channels = max(1, in_channels // 8)
        self.query = nn.Conv2d(in_channels, mid_channels, kernel_size=1)
        self.key = nn.Conv2d(in_channels, mid_channels, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))  # 可学习的权重参数
        
    def forward(self, x):
        batch_size, C, H, W = x.size()
        
        # 计算降采样后的空间尺寸
        h, w = max(1, H // self.reduction_ratio), max(1, W // self.reduction_ratio)
        
        # 生成查询、键、值
        q = self.query(x)
        # 对键和值进行空间降采样以节省内存
        k = F.adaptive_avg_pool2d(self.key(x), (h, w))
        v = F.adaptive_avg_pool2d(self.value(x), (h, w))
        
        # 重塑张量以计算注意力
        q = q.view(batch_size, -1, H * W).permute(0, 2, 1)  # B x (H*W) x C'
        k = k.view(batch_size, -1, h * w)  # B x C' x (h*w)
        v = v.view(batch_size, -1, h * w)  # B x C x (h*w)
        
        # 计算注意力权重
        attn = torch.bmm(q, k)  # B x (H*W) x (h*w)
        attn = F.softmax(attn, dim=-1)
        
        # 应用注意力
        out = torch.bmm(v, attn.permute(0, 2, 1))  # B x C x (H*W)
        out = out.view(batch_size, C, H, W)
        
        # 残差连接
        return self.gamma * out + x

class EfficientCrossAttention(nn.Module):
    """
    内存高效的交叉注意力模块，通过空间降采样减少内存占用
    
    参数:
        in_channels: 输入特征通道数
        reduction_ratio: 空间降采样比例，默认为8
    """
    def __init__(self, in_channels, reduction_ratio=8):
        super().__init__()
        self.reduction_ratio = reduction_ratio
        # 确保通道数至少为8，避免除以8后为0
        mid_channels = max(1, in_channels // 8)
        self.query_conv = nn.Conv2d(in_channels, mid_channels, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, mid_channels, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x, y):
        """
        x: 第一个特征图 (例如：图像特征)
        y: 第二个特征图 (例如：掩码特征)
        """
        batch_size, C, H, W = x.size()
        
        # 计算降采样后的空间尺寸
        h, w = max(1, H // self.reduction_ratio), max(1, W // self.reduction_ratio)
        
        # x作为查询，y作为键和值
        q = self.query_conv(x)
        # 对键和值进行空间降采样以节省内存
        k = F.adaptive_avg_pool2d(self.key_conv(y), (h, w))
        v = F.adaptive_avg_pool2d(self.value_conv(y), (h, w))
        
        # 重塑张量以计算注意力
        q = q.view(batch_size, -1, H * W).permute(0, 2, 1)  # B x (H*W) x C'
        k = k.view(batch_size, -1, h * w)  # B x C' x (h*w)
        v = v.view(batch_size, -1, h * w)  # B x C x (h*w)
        
        # 计算注意力权重
        attn = torch.bmm(q, k)  # B x (H*W) x (h*w)
        attn = F.softmax(attn, dim=-1)
        
        # 应用注意力
        out = torch.bmm(v, attn.permute(0, 2, 1))  # B x C x (H*W)
        out = out.view(batch_size, C, H, W)
        
        # 残差连接
        return self.gamma * out + x

class LinearSelfAttention(nn.Module):
    """
    线性自注意力模块，复杂度为O(N)而非O(N²)，大幅降低内存占用
    
    参数:
        in_channels: 输入特征通道数
    """
    def __init__(self, in_channels):
        super().__init__()
        # 确保通道数至少为8，避免除以8后为0
        mid_channels = max(1, in_channels // 8)
        self.query = nn.Conv2d(in_channels, mid_channels, kernel_size=1)
        self.key = nn.Conv2d(in_channels, mid_channels, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        batch_size, C, H, W = x.size()
        
        # 生成查询、键、值
        q = self.query(x).view(batch_size, -1, H * W)  # B x C' x (H*W)
        k = self.key(x).view(batch_size, -1, H * W)  # B x C' x (H*W)
        v = self.value(x).view(batch_size, -1, H * W)  # B x C x (H*W)
        
        # 对键应用softmax进行归一化
        k_softmax = F.softmax(k, dim=-1)
        
        # 计算上下文向量（线性注意力）
        context = torch.bmm(v, k_softmax.transpose(1, 2))  # B x C x C'
        out = torch.bmm(context, q)  # B x C x (H*W)
        
        # 重塑并应用残差连接
        out = out.view(batch_size, C, H, W)
        return self.gamma * out + x

class LinearCrossAttention(nn.Module):
    """
    线性交叉注意力模块，复杂度为O(N)而非O(N²)，大幅降低内存占用
    
    参数:
        in_channels: 输入特征通道数
    """
    def __init__(self, in_channels):
        super().__init__()
        # 确保通道数至少为8，避免除以8后为0
        mid_channels = max(1, in_channels // 8)
        self.query_conv = nn.Conv2d(in_channels, mid_channels, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, mid_channels, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x, y):
        """
        x: 第一个特征图 (例如：图像特征)
        y: 第二个特征图 (例如：掩码特征)
        """
        batch_size, C, H, W = x.size()
        
        # x作为查询，y作为键和值
        q = self.query_conv(x).view(batch_size, -1, H * W)  # B x C' x (H*W)
        k = self.key_conv(y).view(batch_size, -1, H * W)  # B x C' x (H*W)
        v = self.value_conv(y).view(batch_size, -1, H * W)  # B x C x (H*W)
        
        # 对键应用softmax进行归一化
        k_softmax = F.softmax(k, dim=-1)
        
        # 计算上下文向量（线性注意力）
        context = torch.bmm(v, k_softmax.transpose(1, 2))  # B x C x C'
        out = torch.bmm(context, q)  # B x C x (H*W)
        
        # 重塑并应用残差连接
        out = out.view(batch_size, C, H, W)
        return self.gamma * out + x

class SimpleMaskEncoder(nn.Module):
    """
    简化的掩码编码器，用于提取mask的边界先验信息
    参数量远少于标准encoder，仅用于提取空间边界特征
    """
    def __init__(self, in_channels=1, target_channels=None):
        super().__init__()
        
        # 如果提供了目标通道数，使用它；否则使用默认值
        if target_channels is None:
            target_channels = [0, 16, 32, 64, 128, 256]  # 默认通道配置
        
        self.target_channels = target_channels
        
        # 动态构建卷积层
        self.convs = nn.ModuleList()
        
        # 第0层：直接返回输入（占位）
        self.convs.append(nn.Identity())
        
        # 第1-5层：根据目标通道数构建
        prev_channels = in_channels
        for i in range(1, len(target_channels)):
            current_channels = target_channels[i]
            
            if current_channels == 0:
                # 如果目标通道数为0，添加一个占位层
                self.convs.append(nn.Identity())
            else:
                # 构建实际的卷积层
                stride = 2 if i > 1 else 1  # 第1层不降采样，其余层降采样
                conv = nn.Sequential(
                    nn.Conv2d(prev_channels, current_channels, 3, stride=stride, padding=1),
                    nn.BatchNorm2d(current_channels),
                    nn.ReLU(inplace=True)
                )
                self.convs.append(conv)
                prev_channels = current_channels
        
        # 定义输出通道数
        self.out_channels = target_channels
        
    def forward(self, x):
        # 动态执行前向传播
        features = [x]  # 第0层是原始输入
        current_x = x
        
        for i in range(1, len(self.convs)):
            if self.target_channels[i] == 0:
                # 为0通道层创建零张量占位
                # 计算应有的空间尺寸（根据是否降采样）
                if i == 1:
                    # 第1层不降采样
                    h, w = current_x.shape[2], current_x.shape[3]
                else:
                    # 其他层降采样
                    h, w = current_x.shape[2] // 2, current_x.shape[3] // 2
                
                zero_feature = torch.zeros(current_x.shape[0], 0, h, w, 
                                         device=current_x.device, dtype=current_x.dtype)
                features.append(zero_feature)
            else:
                # 正常的卷积层
                current_x = self.convs[i](current_x)
                features.append(current_x)
        
        return features

class ChannelAdapter(nn.Module):
    """
    通道适配器，将简化mask特征的通道数调整为与主干特征一致
    """
    def __init__(self, mask_channels, img_channels):
        super().__init__()
        self.adapters = nn.ModuleList()
        
        for mc, ic in zip(mask_channels, img_channels):
            if mc == 0 or ic == 0:  # 跳过通道数为0的层（如MiT-B0的第1层）
                self.adapters.append(nn.Identity())
            elif mc != ic:
                # 使用1x1卷积调整通道数
                self.adapters.append(
                    nn.Sequential(
                        nn.Conv2d(mc, ic, 1),
                        nn.BatchNorm2d(ic),
                        nn.ReLU(inplace=True)
                    )
                )
            else:
                self.adapters.append(nn.Identity())
    
    def forward(self, mask_features):
        adapted_features = []
        for feat, adapter in zip(mask_features, self.adapters):
            adapted_features.append(adapter(feat))
        return adapted_features

class DualEncoderUNetPP(nn.Module):
    """
    细粒度分类 FPN 模型（林草数据集适配）：
    - 单分支图像编码器提取 RGB 特征
    - FPN (Feature Pyramid Network) 解码器进行多尺度特征融合
    - 新增：支持同时预测一级、二级和三级标签（细粒度分类）
    - 新增：集成类别关系GNN增强细粒度分类
    - 新增：支持CLIP引导的语义图神经网络
    """
    def __init__(self,
                 encoder_name: str = 'mit_b2',
                 encoder_weights: str = 'imagenet',
                 num_classes_level1: int = 2,  # 一级类别数（林草数据集：0背景 + 1林草 = 2类）
                 num_classes_level2: int = 6,  # 二级类别数（林草数据集：0背景 + 5林草类别 = 6类）
                 num_classes_level3: int = 9,  # 三级类别数（林草数据集：0背景 + 8林草子类 = 9类）
                 use_adaptive_gnn: bool = False,  # 使用基础自适应图神经网络
                 use_clip_gnn: bool = True,  # 使用CLIP引导的图神经网络
                 use_clip_text: bool = True,  # 使用样本级CLIP文本引导（新增）
                 clip_text_device: str = 'cuda:0'):  # CLIP文本编码器设备
        super().__init__()
        
        # 标记是否使用文本引导
        self.use_clip_text = use_clip_text
        # 设置 decoder channels
        decoder_channels = (256, 128, 64, 32, 16)
        
        # RGB 图像 Encoder
        self.enc_img = get_encoder(
            encoder_name,
            in_channels=3,
            depth=5,
            weights=encoder_weights
        )
        
        # 获取编码器的输出通道数
        img_channels = self.enc_img.out_channels
        
        # 检查输出通道数是否为列表形式
        if not isinstance(img_channels, (list, tuple)):
            img_channels = [img_channels]
        
        # 为解码器过滤掉0通道的层
        # 对于某些编码器（如MiT系列），第1层可能是0通道，需要过滤掉
        filtered_encoder_channels = []
        self.valid_layer_indices = []  # 记录有效层的索引
        
        for i, ch in enumerate(self.enc_img.out_channels):
            if ch > 0:  # 只保留通道数大于0的层
                filtered_encoder_channels.append(ch)
                self.valid_layer_indices.append(i)
        
        # 使用FPN解码器，对MiT系列编码器有更好的兼容性
        print(f"使用FPN解码器（Feature Pyramid Network）适配 {encoder_name}")
        
        # FPN解码器配置
        # pyramid_channels：金字塔特征通道数，通常为256
        # segmentation_channels：分割特征通道数，可以设置为128或256
        self.decoder = FPNDecoder(
            encoder_channels=tuple(filtered_encoder_channels),
            encoder_depth=len(filtered_encoder_channels),
            pyramid_channels=256,
            segmentation_channels=128,
            dropout=0.2,
            merge_policy='add'  # 或 'cat'
        )
        
        # FPN解码器的输出通道数由segmentation_channels参数决定
        decoder_output_channels = 128  # 与上面FPN配置的segmentation_channels一致
        
        # 动态检测解码器的实际输出通道数（验证）
        with torch.no_grad():
            try:
                # 创建正确空间尺寸的测试特征
                # MIT编码器的下采样倍数通常是: [1, 4, 8, 16, 32]（对应stride [4, 2, 2, 2, 2]）
                test_features = []
                base_size = 128  # 基础尺寸
                
                for i, ch in enumerate(filtered_encoder_channels):
                    # 根据编码器层级计算下采样倍数
                    # 对于MIT系列：层0通常被过滤，实际使用的是层1-5
                    original_layer_idx = self.valid_layer_indices[i]
                    
                    # MIT系列的实际下采样倍数
                    if original_layer_idx == 0:
                        downsample_factor = 1
                    elif original_layer_idx == 1:
                        downsample_factor = 4  # 1/4
                    elif original_layer_idx == 2:
                        downsample_factor = 8  # 1/8
                    elif original_layer_idx == 3:
                        downsample_factor = 16  # 1/16
                    else:  # original_layer_idx >= 4
                        downsample_factor = 32  # 1/32
                    
                    h = w = base_size // downsample_factor
                    h = max(h, 2)  # 确保最小尺寸
                    w = max(w, 2)
                    test_feat = torch.zeros(1, ch, h, w)
                    test_features.append(test_feat)
                
                # 测试解码器输出
                test_output = self.decoder(*test_features)
                actual_channels = test_output.shape[1]
                
                if actual_channels != decoder_output_channels:
                    print(f"FPN解码器输出通道数调整: {decoder_output_channels} -> {actual_channels}")
                    decoder_output_channels = actual_channels
                else:
                    print(f"✓ FPN解码器输出通道数确认: {decoder_output_channels}")
                    
            except Exception as e:
                print(f"警告: 解码器自动测试失败 ({e})")
                print(f"使用默认FPN输出通道数: {decoder_output_channels}")
        
        # 一级分割头（新增）
        self.seg_head_level1 = nn.Conv2d(
            in_channels=decoder_output_channels,
            out_channels=num_classes_level1,
            kernel_size=1
        )
        
        # 二级分割头
        self.seg_head_level2 = nn.Conv2d(
            in_channels=decoder_output_channels,
            out_channels=num_classes_level2,
            kernel_size=1
        )
        
        # 三级分割头
        self.seg_head_level3 = nn.Conv2d(
            in_channels=decoder_output_channels,
            out_channels=num_classes_level3,
            kernel_size=1
        )
        
        # 新增：基于类别关系的图神经网络，用于增强三级类别预测
        self.use_adaptive_gnn = use_adaptive_gnn
        self.use_clip_gnn = use_clip_gnn
        
        # 初始化GNN模块（平行关系）
        self.adaptive_gnn = None
        self.clip_gnn = None
        
        if use_adaptive_gnn:
            # 使用基础自适应语义图神经网络
            self.adaptive_gnn = AdaptiveSemanticGNN(num_classes=num_classes_level3)
            
        if use_clip_gnn:
            # 使用CLIP引导的语义图神经网络
            self.clip_gnn = CLIPGuidedSemanticGNN(
                num_classes=num_classes_level3,
                hidden_dim=128,
                clip_model_name='ViT-B/32',
                temperature=0.1,
                distill_weight=0.3,
                dataset_type='forest'  # 适配林草数据集
            )
        
        # 检查参数组合的合理性
        if use_adaptive_gnn and use_clip_gnn:
            print("Warning: 同时启用两个GNN模块，将使用融合策略")
        elif not use_adaptive_gnn and not use_clip_gnn:
            print("Info: 未启用任何GNN模块，仅使用基础网络")
        
        # 新增：CLIP文本引导模块（方案1）
        if use_clip_text and CLIP_AVAILABLE:
            print("Info: 启用CLIP样本级文本引导模块（方案1）")
            # 文本编码器
            self.text_encoder = CLIPTextEncoder(
                clip_model_name='ViT-B/32',
                device=clip_text_device
            )
            
            # 为解码器输出添加文本引导注意力
            # decoder_output_channels 是解码器的最终输出通道数
            self.text_guided_attention = CLIPTextGuidedAttention(
                visual_dim=decoder_output_channels,
                text_dim=512,
                hidden_dim=256,
                num_heads=4,
                dropout=0.1
            )
            print(f"文本引导注意力模块初始化完成，视觉维度={decoder_output_channels}")
        else:
            self.text_encoder = None
            self.text_guided_attention = None
            if use_clip_text and not CLIP_AVAILABLE:
                print("Warning: CLIP不可用，文本引导功能已禁用")
        
        # 使用最简单的相加融合方法
        
        # 兼容旧代码
        self.seg_head = self.seg_head_level2

    def forward(self, x: torch.Tensor, 
                text_list: list = None, return_clip_loss: bool = False) -> tuple:
        """
        前向传播（支持文本引导）
        
        参数:
            x: RGB 图像张量，shape = [B,3,H,W]
            text_list: 文本描述列表（可选），长度为B，每个元素是一个字符串
            return_clip_loss: 是否返回CLIP蒸馏损失
        
        返回: 
            - 如果 return_clip_loss=False: (level1_logits, level2_logits, level3_logits)
            - 如果 return_clip_loss=True: (level1_logits, level2_logits, level3_logits, clip_loss)
        """
        # 图像编码器特征提取
        feats_img = self.enc_img(x)
        
        # 只将有效的特征层传递给解码器（过滤掉0通道的层）
        valid_feats = [feats_img[i] for i in self.valid_layer_indices]
        
        # FPN Decoder 解码
        # FPN decoder 直接返回融合后的特征图
        d = self.decoder(*valid_feats)
        
        # 新增：应用CLIP文本引导注意力（在分割头之前）
        if self.use_clip_text and self.text_encoder is not None and text_list is not None:
            # text_list 可以是：
            # 1. 字符串列表（单GPU或推理时）
            # 2. Tensor（多GPU训练时，已预编码）
            
            if isinstance(text_list, torch.Tensor):
                # 情况2：已经是编码好的tensor（多GPU训练）
                text_features = text_list  # [B, 512]
            else:
                # 情况1：字符串列表，需要编码
                # 处理DataParallel情况：可能未正确分割
                current_batch_size = d.shape[0]
                
                if isinstance(text_list, (list, tuple)) and len(text_list) > current_batch_size:
                    # 只取当前batch对应的文本
                    text_list = text_list[:current_batch_size]
                
                # 编码文本
                text_features = self.text_encoder.encode_texts(text_list)  # [B, 512]
            
            # 应用文本引导注意力增强解码器特征
            d = self.text_guided_attention(d, text_features)  # [B, C, H, W]
            
        # 获取输入的空间尺寸
        input_h, input_w = x.shape[2], x.shape[3]
        
        # 一级、二级和三级分割头
        level1_logits = self.seg_head_level1(d)
        level2_logits = self.seg_head_level2(d)
        level3_logits = self.seg_head_level3(d)
        
        # 上采样到输入尺寸
        if level1_logits.shape[2] != input_h or level1_logits.shape[3] != input_w:
            level1_logits = F.interpolate(level1_logits, size=(input_h, input_w), 
                                        mode='bilinear', align_corners=False)
            level2_logits = F.interpolate(level2_logits, size=(input_h, input_w), 
                                        mode='bilinear', align_corners=False)
            level3_logits = F.interpolate(level3_logits, size=(input_h, input_w), 
                                        mode='bilinear', align_corners=False)
        
        # 使用基于类别关系的GNN增强三级类别预测
        clip_loss = 0.0
        
        # 平行处理：可以同时使用两个GNN模块
        if self.use_adaptive_gnn and self.use_clip_gnn:
            # 同时使用两个GNN：融合策略
            adaptive_output = self.adaptive_gnn(level3_logits)
            clip_output = self.clip_gnn(level3_logits)
            # 加权融合两个输出（可以调整权重）
            level3_logits = 0.4 * adaptive_output + 0.6 * clip_output
            # 计算CLIP蒸馏损失
            if hasattr(self.clip_gnn, 'compute_distillation_loss'):
                clip_loss = self.clip_gnn.compute_distillation_loss()
                
        elif self.use_adaptive_gnn:
            # 仅使用基础自适应GNN
            level3_logits = self.adaptive_gnn(level3_logits)
            
        elif self.use_clip_gnn:
            # 仅使用CLIP引导GNN
            level3_logits = self.clip_gnn(level3_logits)
            # 计算CLIP蒸馏损失
            if hasattr(self.clip_gnn, 'compute_distillation_loss'):
                clip_loss = self.clip_gnn.compute_distillation_loss()
        
        # 如果都不使用，level3_logits保持原样
        
        # 根据参数决定是否返回CLIP损失
        if return_clip_loss:
            return level1_logits, level2_logits, level3_logits, clip_loss
        else:
            return level1_logits, level2_logits, level3_logits


if __name__ == '__main__':
    # 简单测试网络前向
    print("测试细粒度分类模型...")
    model = DualEncoderUNetPP(
        encoder_name='mit_b2',  # 使用MiT-B2编码器
        encoder_weights='imagenet',
        num_classes_level1=2,  # 一级类别数
        num_classes_level2=6,  # 二级类别数
        num_classes_level3=9,  # 三级类别数
        use_adaptive_gnn=False,  # 不使用基础GNN
        use_clip_gnn=True       # 使用CLIP GNN
    )
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'总参数量: {total_params:,}')
    print(f'可训练参数量: {trainable_params:,}')
    
    # 使用较小的输入尺寸进行测试
    img = torch.randn(2, 3, 512, 512)  # 降低测试分辨率
    
    # 使用torch.cuda.amp进行混合精度计算以进一步节省内存
    with torch.cuda.amp.autocast(enabled=True):
        level1_out, level2_out, level3_out = model(img)
    
    print('Level1 output shape:', level1_out.shape)  # 预期 [2,2,512,512]
    print('Level2 output shape:', level2_out.shape)  # 预期 [2,6,512,512]
    print('Level3 output shape:', level3_out.shape)  # 预期 [2,9,512,512]
    print('细粒度分类模型：FPN解码器，单分支编码器，预测一级、二级、三级标签')