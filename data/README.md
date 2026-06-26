# 数据说明

本目录包含论文《融合经响应校准 SWAT 农业面源过程信息的闸控浅水湖泊总氮预测》复现所需的最小原始数据集（对应正文 2.2 节）。

## 目录结构

| 路径 | 内容 | 论文用途 |
|------|------|----------|
| `meteorology/气象数据_2023合并.xlsx` | CMFD 气象驱动（降雨、气温、风速、辐射） | SWAT 输入与 TN 预测气象变量 |
| `water_quality/2021-2025水质日平均值结果.xlsx` | 排水闸等地表水自动监测 TN 日均值 | 监督目标 |
| `swat/SWAT_2023_Final_Data.xlsx` | SWAT 日尺度河道/子流域氮输出 | 农业面源过程先验 |
| `hydrology/水文数据.xlsx` | 新滩口、新堤、坪坊、挖沟咀闸控水文 | 水动力代理变量 |
| `hydrology_archive/` | 国家水文数据库标准导出（10 站 × 28 表） | 福田寺率定、闸控背景 |

## 公开数据集引用

- 气象：He J, Yang K, Tang W, et al. (2020) CMFD — *Scientific Data* 7:25
- DEM：Farr et al. (2007) SRTM — *Reviews of Geophysics* 45:RG2004
- 土壤：FAO/IIASA HWSD v1.2
- 土地利用：Liu et al. (2014) CNLUCC — *Journal of Geographical Sciences* 24(2):195-210

## 使用说明

- 建模中间表见 `../outputs/processed/`；无需重新运行 ArcSWAT 即可复现正文表 2 与图 4–7。
- `extensions/` 为可选扩展数据占位目录（本仓库未包含额外监测资料）。
- 水文站数据来源于国家水文数据库导出格式；请遵守数据提供方使用规定。
