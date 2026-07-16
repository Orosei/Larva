import torch
import torch.nn as nn


# 输入输出相同
class SElayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SElayer,self).__init__()
        # 自适应平均池化层，将每个通道的空间纬度（H,W）压缩到1*1
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        # 全连接层序列，包含两个线性变换和中间的ReLU激活函数
        self.fc = nn.Sequential(
            # 第一个线性层，将channel1缩减到channel1//reduction
            nn.Linear(channel, channel // reduction, bias=False),
            # 激活ReLU函数，引入非线性
            nn.ReLU(inplace = True),
            # 第二个线性层，将channel1 // reduction恢复到channel1
            nn.Linear(channel // reduction, channel, bias=False),
            # 激活Sigmoid，将输出限制在（0,1）之间
            nn.Sigmoid()
            )
    def forward(self, x):
        # 获取输入张量的形状
        b, c, h, w = x.size()
        # squeeze：通过全局平均池化层，将每个通道的空间纬度（H,W）压缩到1*1
        y = self.avg_pool(x).view(b, c)
        # excitation：通过全连接层序列，对压缩后的特征进行处理
        y = self.fc(y).view(b, c, 1, 1)
        # 通过扩展后的注意力权重y调整输入张量x的每个通道
        return x * y.expand_as(x)

class EmbedLayer(nn.Module):
    def __init__(self, n_channels, embed_dim, im_size, patch_size, dropout=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(n_channels, embed_dim, kernel_size=patch_size, stride=patch_size)  # Patch Encoding
        self.pos_embedding = nn.Parameter(torch.zeros(1, (im_size // patch_size) ** 2, embed_dim), requires_grad=True)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim), requires_grad=True)  # Classification Token
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B = x.shape[0]
        x = self.conv1(x)  # B, C, IH, IW-> B, E, IH/P, IW/P  Split images into the patches and embed patches
        x = x.reshape([B, x.shape[1], -1])  # B, E, IH/P, IW/P -> B, E, (IH/P*IW/P) -> B, E, N    Flattening the patches
        x = x.permute(0, 2, 1)  # B, E, N -> B, N, E Rearrange to put sequence dimension in the middle
        x = x + self.pos_embedding  # B, N, E-> B, N, E  Add positional embedding
        x = torch.cat((torch.repeat_interleave(self.cls_token, B, 0), x), dim=1)  # B, (N+1), E Add classification token
        x = self.dropout(x)
        return x


class SelfAttention(nn.Module):
    def __init__(self, embed_dim, n_attention_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_attention_heads = n_attention_heads
        self.head_embed_dim = embed_dim // n_attention_heads

        self.queries = nn.Linear(self.embed_dim, self.head_embed_dim * self.n_attention_heads)  # Queries projection
        self.keys = nn.Linear(self.embed_dim, self.head_embed_dim * self.n_attention_heads)  # Keys projection
        self.values = nn.Linear(self.embed_dim, self.head_embed_dim * self.n_attention_heads)  # Values projection
        self.out_projection = nn.Linear(self.head_embed_dim * self.n_attention_heads, self.embed_dim)  # Out projection

    def forward(self, x):
        b, s, e = x.shape  # Note: In case of self-attention Q, K and V are all equal to S
        xq = self.queries(x).reshape(b, s, self.n_attention_heads, self.head_embed_dim)  # B, Q, E -> B, Q, H, HE
        xq = xq.permute(0, 2, 1, 3)  # B, Q, H, HE  ->  B, H, Q, HE
        xk = self.keys(x).reshape(b, s, self.n_attention_heads, self.head_embed_dim)  # B, K, E ->  B, K, H, HE
        xk = xk.permute(0, 2, 1, 3)  # B, K, H, HE  ->  B, H, K, HE
        xv = self.values(x).reshape(b, s, self.n_attention_heads, self.head_embed_dim)  # B, V, E-> B, V, H, HE
        xv = xv.permute(0, 2, 1, 3)  # B, V, H, HE  ->  B, H, V, HE
        # Compute Attention pre-softmax values
        xk = xk.permute(0, 1, 3, 2)  # B, H, K, HE  ->  B, H, HE, K
        x_attention = torch.matmul(xq, xk)  # B, H, Q, HE* B, H, HE, K ->  B, H, Q, K
        x_attention /= float(self.head_embed_dim) ** 0.5  # Scale pre-softmax values for stability
        x_attention = torch.softmax(x_attention, dim=-1)  # Compute Attention Matrix
        x = torch.matmul(x_attention, xv)  # B, H, Q, K  *  B, H, V, HE  ->  B, H, Q, HE

        # Format the output
        x = x.permute(0, 2, 1, 3)  # B, H, Q, HE -> B, Q, H, HE
        x = x.reshape(b, s, e)  # B, Q, H, HE -> B, Q, (H*HE)
        x = self.out_projection(x)  # B, Q,(H*HE) -> B, Q, E
        return x


class Encoder(nn.Module):
    def __init__(self, embed_dim, n_attention_heads, forward_mul, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attention = SelfAttention(embed_dim, n_attention_heads)
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(embed_dim)
        self.fc1 = nn.Linear(embed_dim, embed_dim * forward_mul)
        self.activation = nn.GELU()
        self.fc2 = nn.Linear(embed_dim * forward_mul, embed_dim)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.dropout1(self.attention(self.norm1(x)))  # Skip connections
        x = x + self.dropout2(self.fc2(self.activation(self.fc1(self.norm2(x)))))  # Skip connections
        return x


class Classifier(nn.Module):
    def __init__(self, embed_dim, n_classes):
        super().__init__()
        # New architectures skip fc1 and activations and directly apply fc2.
        self.fc1 = nn.Linear(embed_dim, embed_dim)
        self.activation = nn.Tanh()
        self.fc2 = nn.Linear(embed_dim, n_classes)

    def forward(self, x):
        x = x[:, 0, :]  # B, S, E -> B, E          Get CLS token
        x = self.fc1(x)  # B, E    -> B, E
        x = self.activation(x)  # B, E    -> B, E
        x = self.fc2(x)  # B, E    -> B, CL
        return x


class VisionTransformer(nn.Module):
    def __init__(self, n_channels, embed_dim, n_layers, n_attention_heads, forward_mul, image_size, patch_size,
                 n_classes, dropout=0.1):
        super().__init__()
        self.embedding = EmbedLayer(n_channels, embed_dim, image_size, patch_size, dropout=dropout)
        self.encoder = nn.ModuleList(
            [Encoder(embed_dim, n_attention_heads, forward_mul, dropout=dropout) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(embed_dim)  # Final normalization layer after the last block
        self.classifier = Classifier(embed_dim, n_classes)
        self.apply(vit_init_weights)  # Weight initialization

    def forward(self, x):
        x = self.embedding(x)
        for block in self.encoder:
            x = block(x)
        x = self.norm(x)
        x = self.classifier(x)
        return x


def vit_init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, mean=0.0, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

    elif isinstance(m, EmbedLayer):
        nn.init.trunc_normal_(m.cls_token, mean=0.0, std=0.02)
        nn.init.trunc_normal_(m.pos_embedding, mean=0.0, std=0.02)

class DirectionNet(nn.Module):
    def __init__(self):
        super(DirectionNet, self).__init__()
        self.apply(dn_init_weights)

        # Initial convolution block
        self.initial_conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        # Multi-scale convolution blocks
        self.conv_3x3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.conv_5x5 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.conv_7x7 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=7, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # Main body of the network with alternating dwConv and transition blocks
        self.senet = SElayer(channel=96)
        self.dwConv1 = self._create_dwConv_block(32 * 3, 7)
        self.transition1 = self._create_transition_block(32 * 3, 64, 9)
        self.senet1 = SElayer(channel=64)
        self.dwConv2 = self._create_dwConv_block(64, 5)
        self.transition2 = self._create_transition_block(64, 128, 5)
        self.senet2 = SElayer(channel=128)
        self.dwConv3 = self._create_dwConv_block(128, 3)
        self.transition3 = self._create_transition_block(128, 128, 3)
        self.senet3 = SElayer(channel=128)

        # Adaptive average pooling layer
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.mlp = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16)
        )
        # Final 1x1 convolution layer
        #self.final_conv = nn.Conv2d(128, 40, kernel_size=1, padding=0)
        self.final_conv = nn.Conv2d(128, 16, kernel_size=1, padding=0)

        #self.fc = nn.Linear(40, 40)
        self.fc = nn.Linear(16, 16)

    def _create_dwConv_block(self, in_channels, kernel_size):
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.Identity()
        )

    def _create_transition_block(self, in_channels, out_channels, kernel_size):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.initial_conv(x)
        # Multi-scale feature extraction
        x_3x3 = self.conv_3x3(x)
        x_5x5 = self.conv_5x5(x)
        x_7x7 = self.conv_7x7(x)
        # Feature fusion by concatenation
        x_fused = torch.cat([x_3x3, x_5x5, x_7x7], dim=1)
        x_fused = self.senet(x_fused)
        x = self.dwConv1(x_fused)
        x = self.transition1(x)
        x = self.dwConv2(x)
        x = self.transition2(x)
        x = self.dwConv3(x)
        x = self.transition3(x)
        x = self.adaptive_pool(x)
        # x = self.final_conv(x)
        #print("final_conv output:", x.mean(), x.std())
        x = x.view(x.size(0), -1)
        #print("view output:", x.mean(), x.std())
        # x = self.fc(x)
        x = self.mlp(x)
        #print("fc output:", x.mean(), x.std())
        x = torch.sigmoid(x)
        #print("sigmoid output:", x.mean(), x.std())
        return x

def dn_init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, mean=0.0, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


# class DirectionNet(nn.Module):
#     def __init__(self):
#         super(DirectionNet, self).__init__()
#
#         # Initial convolution block
#         self.initial_conv = nn.Sequential(
#             nn.Conv2d(3, 32, kernel_size=3),
#             nn.BatchNorm2d(32),
#             nn.ReLU()
#         )
#
#         # Main body of the network with alternating dwConv and transition blocks
#         self.dwConv1 = self._create_dwConv_block(32, 7)
#         self.transition1 = self._create_transition_block(32, 64, 9)
#         self.dwConv2 = self._create_dwConv_block(64, 7)
#         self.transition2 = self._create_transition_block(64, 128, 9)
#         self.dwConv3 = self._create_dwConv_block(128, 7)
#         self.transition3 = self._create_transition_block(128, 128, 7)
#
#         # Adaptive average pooling layer
#         self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
#
#         # Final 1x1 convolution layer
#         #self.final_conv = nn.Conv2d(128, 40, kernel_size=1, padding=0)
#         self.final_conv = nn.Conv2d(128, 8, kernel_size=1, padding=0)
#
#         self.fc = nn.Linear(8, 8)
#
#     def _create_dwConv_block(self, in_channels, kernel_size):
#         return nn.Sequential(
#             nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, groups=in_channels),
#             nn.BatchNorm2d(in_channels),
#             nn.ReLU(),
#             nn.Conv2d(in_channels, in_channels, kernel_size=1),
#             nn.BatchNorm2d(in_channels),
#             nn.Identity()
#         )
#
#     def _create_transition_block(self, in_channels, out_channels, kernel_size):
#         return nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=1),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(),
#             nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU()
#         )
#
#     def forward(self, x):
#         x = self.initial_conv(x)
#         x = self.dwConv1(x)
#         x = self.transition1(x)
#         x = self.dwConv2(x)
#         x = self.transition2(x)
#         x = self.dwConv3(x)
#         x = self.transition3(x)
#         x = self.adaptive_pool(x)
#         x = self.final_conv(x)
#         x = x.view(x.size(0), -1)
#         x = self.fc(x)
#
#         return x