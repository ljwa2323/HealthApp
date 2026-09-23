# HealthApp 系统架构

## 1. 产品定位

HealthApp 不是一次问诊聊天工具，而是一个患者拥有的纵向健康档案系统。

核心目标是把分散在医院报告、检查单、处方、出院记录、体检单、手机截图和患者自述中的信息，转换成一条可追溯、可计算、可供 LLM/Agent 使用的健康时间轴。

## 2. 核心原则

### 2.1 原始证据永久保留

任何结构化结果都不能替代原始记录。系统始终保留：

- 原始图片/PDF/文件
- OCR 原文
- OCR 版面块及页码/坐标
- 文档元数据
- 每个标准化事实对应的证据位置

这样可以做到从一个结构化事实反向定位到原始报告。

### 2.2 时间优先

医疗记录至少存在三类时间：

- `event_time`: 事件真正发生的时间，如抽血时间、影像检查时间、服药开始时间
- `document_time`: 医疗文档形成或签发时间
- `ingested_at`: 用户上传到 HealthApp 的时间

LLM 推理默认按照 `event_time` 排序，绝不能把上传时间误当作临床时间。

### 2.3 事实和 AI 推断分离

系统数据分三层：

1. Evidence Layer：原始文件、OCR、原文证据
2. Canonical Fact Layer：经过标准化的临床事实
3. Derived Intelligence Layer：趋势、摘要、风险提示、Agent 推断

AI 产生的内容必须记录模型、时间、输入范围和依据，不能覆盖第二层。

## 3. 总体架构

```text
Patient upload / camera / manual input
                |
                v
        Document ingestion
                |
     +----------+----------+
     |                     |
     v                     v
 OCR / parser        Metadata extraction
     |                     |
     +----------+----------+
                v
        Document classifier
                |
                v
          LLM normalizer
                |
                v
      Validation / confirmation
                |
                v
        Canonical event store
                |
     +----------+-----------+----------------+
     |                      |                |
     v                      v                v
 Timeline views        Trend engine     Agent context
     |                      |                |
     +----------------------+----------------+
                            v
                    Patient-facing insight
```

## 4. OCR 与 LLM 的分工

默认使用 OCR，而不是让视觉语言模型直接解释医学图像。

推荐流程：

1. 图片质量检查：方向、模糊、曝光、裁剪
2. OCR：输出文字及版面坐标
3. 文档类型识别：检验、影像报告、病理、处方、门诊、出院记录等
4. 规则 + LLM 抽取：抽取日期、指标、数值、单位、参考范围、诊断、用药
5. 单位和术语标准化
6. 冲突检测和低置信度复核
7. 写入 canonical timeline

对于 CT/MRI/超声等检查，初版只解析正式影像报告文本。原始影像不交给 VL 模型重新诊断。

## 5. 数据存储建议

生产环境建议：

- PostgreSQL：事件、事实、术语、时间关系、审计记录
- Object storage：原始图片/PDF
- PostgreSQL JSONB：文档级灵活结构
- pgvector 或独立向量库：用于语义检索，但不作为事实主存储
- Redis：任务队列、OCR/LLM异步处理缓存

## 6. Agent 使用方式

Agent 不应该一次读取患者全部历史原文，而应通过 Context Builder 获取任务相关上下文。

例如用户问：最近两年肾功能有没有恶化？

Context Builder 应返回：

- 肾功能相关检验时间序列
- 相关诊断和住院事件
- 影响肾功能的药物
- 对应证据引用
- 数据缺口和时间跨度

这样可以降低 token 消耗并减少无关信息干扰。

## 7. 安全与隐私

生产版本至少应包括：

- 传输和存储加密
- 用户级数据隔离
- 敏感操作审计日志
- 原始数据删除与导出
- AI 输出显著标明为辅助信息
- 低置信度 OCR/抽取进入人工确认
- 不将患者原始数据默认用于模型训练
- 所有 Agent 结论保留证据链

## 8. MVP 优先级

第一阶段重点不是做复杂诊断 Agent，而是把纵向健康数据基础设施做稳：

1. 图片/PDF 上传
2. OCR 与文档分类
3. 结构化抽取
4. 可编辑时间轴
5. 检验趋势
6. 药物与诊断纵向视图
7. Agent context API

只有这套数据层稳定后，再加入风险预测、个性化提醒和临床辅助推理。
