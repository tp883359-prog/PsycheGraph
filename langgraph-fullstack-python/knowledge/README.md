# 理论知识库（Knowledge Base）

这个目录存放 PsycheGraph 用来做理论检索（RAG）的**本地**文献材料。没有任何材料会被自动
下载；系统只索引你自己放进来的文件。

## 目录结构

```
knowledge/
├── README.md            # 本文件
├── freudian/            # 弗洛伊德精神分析相关材料
├── object_relations/    # 客体关系相关材料
└── lacanian/            # 拉康相关材料
```

- 放在 `freudian/`、`object_relations/`、`lacanian/` 下的文件会被标记为对应学派，
  检索时只会被对应学派的 Specialist 引用。
- 放在其他目录（或 `knowledge/` 根目录）下的文件会被标记为 `general`，三个学派都可检索到，
  适合放方法论、术语表这类共享说明。

## 支持的文件格式

| 格式 | 说明 |
|---|---|
| `.md` | 推荐。可用 `#` 标题，标题会作为 chunk 的 `section` 元数据 |
| `.txt` | 纯文本，无 `section` |
| `.pdf` | 会按真实页码解析，`page` 元数据来自 PDF 本身 |

不会被索引的文件：`README*`（例如本文件）、以 `.` 或 `_` 开头的文件、
以及非 `.md`/`.txt`/`.pdf` 的文件。

## 可选的文件头（front matter）

`.md` / `.txt` 文件可以在开头写一段元数据（`---` 包围），用来记录真实出处：

```markdown
---
title: 梦的解析（项目摘录笔记）
author: 你自己填写的作者
work_title: 作品名
year: 1900
language: zh
school: freudian
---

正文……
```

规则：

- 只写你确实知道的信息。**不知道就留空**，系统会记为 `None`，不会编造页码或年份。
- 不写 front matter 也可以，此时 `title` 取文件名，其余字段为 `None`。
- `school` 字段可覆盖目录归属（一般不需要写）。

## 版权与来源原则

请只放入以下三类材料：

1. **公有领域（public domain）** 的原文；
2. 你拥有使用权或明确许可的文本；
3. **你自己编写的**理论说明、读书笔记、术语整理。

不要为了让检索有数据而从网上下载版权不明的书籍，也不要放入网络摘要或模型生成的内容并
把它当成 Freud / Lacan 的原文。PsycheGraph 的知识库里出现什么，回答就会引用什么。

> `tests/fixtures/knowledge/` 下的测试语料是本项目自编的**测试材料**（明确标注
> `test_fixture: true`），不是 Freud / Lacan 的原文，只用于验证检索机制。

## 建立索引

放好材料后，在项目根目录运行：

```powershell
$env:PYTHONUTF8="1"
uv run python scripts/index_knowledge.py            # 增量写入（chunk_id 稳定，重复运行会覆盖同名 chunk）
uv run python scripts/index_knowledge.py --rebuild  # 先清空集合再重建
```

索引是**离线**步骤：向量数据库写在 `data/vectorstore/`（已 gitignore），
日常运行 Graph 时只读取已有索引，不会重新 embedding。

## 相关环境变量

| 变量 | 默认值 | 说明 |
|---|---|---|
| `EMBEDDING_MODEL` | `BAAI/bge-m3` | 本地 embedding 模型 |
| `EMBEDDING_DEVICE` | `cpu` | 有 GPU 且显式配置时可设为 `cuda` |
| `RAG_VECTORSTORE_PATH` | `data/vectorstore` | 向量库目录 |
| `RAG_COLLECTION` | `psychegraph_theory` | Chroma collection 名称 |
| `RAG_TOP_K` | `5` | 每个学派检索的 chunk 数 |
| `RAG_ENABLED` | `true` | 设为 `false` 可完全关闭检索 |
