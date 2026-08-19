---
layout: post
title: "智能体记忆（Agentic Memory）"
date: 2026-08-19 00:00:00 +0800
categories: [llm, agent, notes]
---

智能体记忆，指的是 AI 智能体（agent）跨越时间存储、检索并使用信息的能力——既包括在单个任务内部，也包括跨越不同对话或会话。

与只"知道"当前对话窗口内容的基础聊天机器人不同，具备记忆的智能体能够随着时间推移，逐步建立起对用户、用户工作以及用户偏好的持久理解。

## 智能体记忆的主要类型

**上下文内记忆（In-context memory）** 是智能体的短期记忆或工作记忆：当前处于上下文窗口内的一切内容，例如系统提示词、最近的对话轮次、工具返回结果以及注入的上下文。它快速、即时可用，但它是临时的，并受模型上下文长度的限制。

**外部/持久记忆（External / persistent memory）** 是存储在模型之外的信息，位于文件、数据库、键值存储或向量存储中。它是智能体的长期记忆层：智能体把有用的事实、偏好、文档和任务状态写入其中，之后只检索相关的片段。正是这一层，让智能体能够跨会话记住你的名字、你的偏好、进行中的项目上下文或过往的参考资料。

**语义记忆（Semantic memory）** 指的是泛化的知识：模型在其权重中已经"知道"的东西，例如通过预训练或微调学到的事实、概念和模式。它不同于外部记忆：基于嵌入（embedding）的语义检索，通常是搜索**外部记忆**的一种方式，而不是语义记忆本身。

**情景记忆（Episodic memory）** 以"事件"的形式捕捉过往经历：智能体或用户当时想要做什么、采取了哪些行动、发生了什么、以及从中学到了什么。它帮助智能体复用成功的模式、避免重蹈覆辙，并基于以往经验（而不仅仅是抽象事实）进行推理。

**程序性记忆（Procedural memory）** 编码的是"如何做事"：技能、工作流、策略，或可复用的任务流程，智能体可以在执行时遵循或检索它们。它更多关乎记住"方法"，而不是记住"事实"。

## 为什么它很重要

没有记忆，每一次交互都从零开始。有了记忆，智能体可以：

- 根据特定用户的角色和专业水平定制回答
- 避免重复用户此前已纠正过的错误
- 在长期运行的项目中保持连续性
- 随时间推移建立起对用户目标更丰富的理解

真正的挑战在于决定*记住什么*、*何时检索*，以及如何在世界不断变化时防止记忆过时。这些都是 AI 智能体领域活跃的研究与工程课题。

## 上下文窗口里有什么

上下文窗口是 LLM 在推理时看到的完整"工作记忆"。下面是一个概念性的提示词组装顺序。典型顺序，自上而下：

1. **系统提示词（System prompt）** —— 行为指令放在最前：模型的角色、规则和语气。从概念上说，这一层应保持精简、聚焦，用户/画像数据应作为独立的一层来处理——即使某些 API 会把这些层序列化并合并进单个顶层 `system` 字段。

2. **用户画像、持久偏好、项目上下文** —— 关于用户的稳定信息，它塑造着每一次回答：用户是谁、长期偏好、当前项目需求、关键决策。把它从系统提示词中抽离出来，是为了让"指令"和"数据"分离，也便于每个会话动态更新。

3. **工具定义（Tool definitions）** —— 结构化的 schema（名称、描述、参数），模型读取它以了解自己能调用哪些工具、以及如何调用。通常在把工具传给模型时，由平台/API 自动注入。

4. **能力注册表 / 程序性上下文（Capability registry / procedural context）** —— 对智能体可用的可复用流程与集成的紧凑描述，例如技能（skills）、操作手册（playbooks）、策略、MCP 服务器或 MCP 资源模板。与工具不同，它们往往是作为提示词侧的指令或元数据注入的，而不是作为可调用的 API schema。

5. **对话历史（Conversation history）** —— 按时间顺序排列的完整来回对话，其中交错包含：
   - 用户消息
   - 助手消息（包括助手发起的任何工具调用）
   - 工具结果（每次工具调用后返回的响应）

6. **当前用户消息** —— 最新的输入，位于最末尾，并在其内部前置了与本次查询相关的 RAG 上下文。把检索到的证据和触发它的查询捆绑在一起，既利用了"近因效应"（recency effect），又让两者在语义上保持关联。

有几个值得知道的细微之处：模型往往更关注上下文最开头和最末尾的内容（即"迷失在中间"，lost in the middle 现象），这正是系统提示词和当前用户消息如此有影响力的原因。随着对话变长，较早的历史会被截断或摘要以腾出空间——这也是持久化外部记忆之所以重要的部分原因。

在许多智能体技术栈中，能力元数据被拆分到多种机制里。可调用的工具可能通过顶层 `tools` 字段到达，而技能注册表、工作流描述或 MCP 集成提示，则可能作为系统提示词或开发者提示词的一部分注入。检索到的 MCP 资源内容一旦加载，其行为就和其他被检索的上下文一样。

## 上下文窗口示例（Anthropic API 格式）

在 Anthropic 的 Messages API 中，`system` 和 `tools` 是顶层字段；把检索到的上下文放进最后一条用户消息内部，是一种应用层模式，而非 API 强制要求的字段。

```json
{
  "system": "You are a helpful customer support agent for Acme Store. Today's date is May 19, 2026. Be concise and friendly.\n\nCapability registry / procedural context:\n- Skill: refund-eligibility-checker — use when deciding whether a delayed order qualifies for a refund.\n- MCP server: policy-docs — use for internal refund-policy and shipping-policy documents.",

  "tools": [
    {
      "name": "search_orders",
      "description": "looks up order status",
      "input_schema": {
        "type": "object",
        "properties": {
          "order_id": { "type": "string" }
        },
        "required": ["order_id"]
      }
    },
    {
      "name": "issue_refund",
      "description": "initiates a refund",
      "input_schema": {
        "type": "object",
        "properties": {
          "order_id": { "type": "string" },
          "amount": { "type": "number" }
        },
        "required": ["order_id", "amount"]
      }
    }
  ],

  "messages": [
    {
      "role": "user",
      "content": "Hi, I placed an order last week but haven't heard anything."
    },
    {
      "role": "assistant",
      "content": "I'd be happy to help. Could you share your order number?"
    },
    {
      "role": "user",
      "content": "It's #A1042."
    },
    {
      "role": "assistant",
      "content": [
        {
          "type": "tool_use",
          "id": "tu_001",
          "name": "search_orders",
          "input": { "order_id": "A1042" }
        }
      ]
    },
    {
      "role": "user",
      "content": [
        {
          "type": "tool_result",
          "tool_use_id": "tu_001",
          "content": "{\"status\": \"delayed\", \"estimated_delivery\": \"2026-05-22\", \"reason\": \"weather\"}"
        }
      ]
    },
    {
      "role": "assistant",
      "content": "Your order #A1042 is delayed due to weather and should arrive by May 22nd. Sorry for the inconvenience!"
    },
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "Acme refund policy: customers may request a full refund if delivery is delayed more than 5 days past the original estimate. Original estimate for #A1042 was May 17."
        },
        {
          "type": "text",
          "text": "Can I get a refund?"
        }
      ]
    }
  ]
}
```

有几点值得注意：

- **工具结果是以 `role: user` 消息发送的**，而不是某个特殊角色。在正常的工具使用流程里，消息通常在 `user` 和 `assistant` 之间交替；不过 Anthropic 也支持一些特殊情况，例如用于受约束续写的末尾 `assistant` 预填充（prefill）。
- **技能和 MCP 提示往往存在于 `system` 或开发者上下文中**，而不在顶层 `tools` 数组里。`tools` 字段描述的是可调用的工具 schema；提示词侧的能力注册表描述的是可复用的流程与集成。
- **RAG 上下文作为一个文本块，前置在最后一条用户消息内部**，与真正的查询并列——这也是大多数系统实际注入它的方式。
- **工具调用作为一个内容块出现在 `assistant` 消息内部**，并在下一条 `user` 消息里有一个对应的 `tool_result` 块——它们始终通过 `tool_use_id` 配对。

## 当上下文窗口满了

当上下文窗口被填满时，智能体需要压缩、丢弃或检索，而不是把整段原始对话一路带下去。常见的解决办法有：

- **摘要旧的轮次。** 用一段关于目标、决策、约束和未决事项的紧凑摘要，替换较早的聊天历史。
- **只保留工作记忆。** 保留最近的若干轮次，加上任何仍然活跃相关的内容，丢弃过时的讨论。
- **把长期事实存到外部。** 把持久信息（如用户偏好、项目需求或重要决策）转移到数据库/向量存储中，之后只检索需要的部分。
- **单独记录情景。** 把重要事件（如"尝试了 X，因为 Y 而失败"）保存到事件日志里，而不是把每一轮都留在提示词中。
- **每轮重建上下文。** 对每个新请求，从头构建一个全新的提示词，自上而下：
   - **系统指令** —— 放最前，因为它定义了模型的行为、边界和任务框架。
   - **稳定的用户/画像记忆** —— 其次，因为持久偏好、用户事实和项目约束，应在模型决定如何行动之前就塑造它的规划。
   - **工具定义** —— 再之后，因为模型需要在被用户与任务上下文"锚定"之后，才知道自己能采取哪些可调用的动作。
   - **能力注册表 / 程序性上下文** —— 一份可用技能、操作手册、MCP 服务器、资源模板或其他可复用流程的紧凑清单。它告诉模型有哪些非工具能力可供加载、查看或遵循。
   - **旧轮次的摘要** —— 在不重放完整历史的前提下，保留来自早前对话的连续性。
   - **检索到的事实/文档/资源** —— 与查询相关的外部知识紧随其后，以便模型拥有当前请求可能需要的证据。这可能包括文件、数据库结果、网页片段或已加载的 MCP 资源。如果任务是知识密集型的，这些通常比过往情景更重要，因此应优先考虑。
   - **相关的过往情景** —— 当以往的成功、失败与教训能够影响规划或能力使用时，把它们放在这里。它们通常排在检索到的文档之后，因为它们是经验性的指引而非首要证据；不过对于高度依赖动作的工作流，你也可以把情景排在文档之前。
   - **最近的轮次** —— 它们应靠近当前请求，因为它们是即时的、局部的对话上下文。
   - **当前用户消息** —— 放最后，好让模型的注意力落在它此刻必须回答的确切请求上。在实现中，与查询相关的检索文档常常被捆绑或前置在这最后一条用户消息内部；上面这份清单是一个概念性的组装顺序，而非强制的线上传输格式。

由于"迷失在中间"效应，靠近提示词末尾的内容可能比埋在中间的内容获得更多注意力。实践中，这意味着最重要的、与本轮相关的内容，往往应放在靠近当前用户消息的位置。至于哪一项配得上这个位置，则取决于任务：如果关键输入是与查询相关的证据，就把检索到的文档紧挨在最后一条用户消息旁边；如果关键输入是即时的对话连续性，就把最近的轮次放在那里；如果关键输入是一次以往的失败或可复用的教训，就把最相关的情景同样放在靠近末尾处。稳定的指令通常仍应放在靠近顶部的位置，因为提示词的开头同样高度显著。所以实用的原则不是始终保持某种僵化的顺序，而是把全局重要的规则放在靠近顶部，同时把对本轮至关重要的证据和上下文移到更靠近当前用户消息的位置。

对于这种动态排序，一个有用的模式是：

- **系统指令**
- **稳定的用户/画像记忆**
- **工具定义**
- **能力注册表 / 程序性上下文**
- **旧轮次的摘要**
- **较低优先级的检索/支撑上下文**
- **最近的轮次**
- **本轮最高优先级的检索文档或情景**
- **当前用户消息**

### Anthropic Messages API 格式示例

下面的示例展示了这种动态排序模式如何映射到 Anthropic 实际的 Messages API 线上格式。

在 Anthropic 的 API 中，它通常变成：

- `系统指令`、`稳定的用户/画像记忆`、`能力注册表 / 程序性上下文` 以及 `旧轮次的摘要` 放进顶层 `system` 字段
- `工具定义` 放进顶层 `tools` 字段
- `最近的轮次` 放进 `messages`
- `较低优先级的检索/支撑上下文`、已加载的资源、`本轮最高优先级的检索文档或情景`，以及 `当前用户消息`，一起捆绑进最后一条 `user` 消息

```json
{
  "model": "claude-sonnet-4-5",
  "max_tokens": 800,
  "system": "You are a helpful travel-planning agent. Follow company policy, respect saved user preferences, and avoid repeating known failures.\n\nStable user/profile memory:\n- User prefers aisle seats.\n- Trips should stay within a strict budget.\n- Team uses a shared approval process for exceptional bookings.\n\nCapability registry / procedural context:\n- Skill: corporate-travel-policy-checker — use when itineraries may violate internal travel policy.\n- Skill: traveler-identity-validation — use before booking to verify legal travel identity.\n- MCP server: travel-docs — provides policy documents and traveler records.\n- MCP resource template: traveler_profile/{traveler_id} — load when traveler identity details are needed.\n\nSummary of older turns:\n- Goal: book compliant travel for the next trip.\n- Decisions: apply budget filters early and preserve booking-related failures as reusable lessons.\n- Constraints: validate traveler identity before purchase.\n- Open question: whether manager approval is needed for this itinerary.",
  "tools": [
    {
      "name": "search_flights",
      "description": "Find available flights that satisfy itinerary and policy constraints.",
      "input_schema": {
        "type": "object",
        "properties": {
          "origin": { "type": "string" },
          "destination": { "type": "string" },
          "date": { "type": "string" }
        },
        "required": ["origin", "destination", "date"]
      }
    },
    {
      "name": "book_flight",
      "description": "Book a selected itinerary after validation checks pass.",
      "input_schema": {
        "type": "object",
        "properties": {
          "itinerary_id": { "type": "string" },
          "traveler_id": { "type": "string" }
        },
        "required": ["itinerary_id", "traveler_id"]
      }
    }
  ],
  "messages": [
    {
      "role": "user",
      "content": "Can you book my next trip to Tokyo for next Thursday?"
    },
    {
      "role": "assistant",
      "content": "Yes. I will use your saved preferences and check policy constraints before booking."
    },
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "Lower-priority retrieved/supporting context:\n- Travel policy: bookings above the budget cap require manager approval.\n- Loaded MCP resource: traveler profile docs say the legal passport name must be validated before purchase."
        },
        {
          "type": "text",
          "text": "Highest-priority retrieved doc or episode for this turn:\n- Previous flight booking failed because the passport name did not match the legal document.\n  Lesson: validate the passport legal name before purchase."
        },
        {
          "type": "text",
          "text": "Please book the next trip to Tokyo for next Thursday, using my saved constraints."
        }
      ]
    }
  ]
}
```

## 具体的摘要 schema

当你摘要旧的轮次时，一份好的结构化摘要只保留将来会有用的信息。一个实用的 schema 是：

- **`goal`（目标）**：用户当前的目标
- **`decisions`（决策）**：对话过程中已经做出的决策
- **`constraints`（约束）**：需求、限制、偏好或不可妥协的条件
- **`durable_facts`（持久事实）**：值得跨会话继续带下去的稳定事实
- **`open_questions`（未决问题）**：尚未解决的问题或缺失的信息
- **`next_step`（下一步）**：智能体接下来应采取的动作
- **`episode_notes`（情景笔记）**：关于重要失败、成功或所学教训的简短记录

示例：

```json
{
  "goal": "Help the user build a travel assistant",
  "decisions": [
    "Use budget filtering early in the workflow",
    "Keep user seat preferences as durable memory"
  ],
  "constraints": [
    "Must support aisle seat preferences",
    "Must stay within a strict budget"
  ],
  "durable_facts": [
    "The team uses a shared approval process"
  ],
  "open_questions": [
    "How should episodic failures be stored and retrieved?"
  ],
  "next_step": "Design the episodic memory storage and retrieval flow",
  "episode_notes": [
    {
      "event": "Previous booking failed because the passport name did not match",
      "outcome": "failure"
    }
  ]
}
```

这类 schema 之所以有用，是因为它把长期存活的事实与临时的工作上下文分离开来，并给了智能体一种紧凑的方式来重建上下文，而无需重放整段对话。

## 外部记忆

外部记忆可以用不同的存储类型来实现，而每一种都擅长承载不同种类的信息。

### 存储类型

- **结构化存储** —— 关系型数据库（如 Postgres 或 SQLite），或字段与索引清晰的文档数据库。
- **向量存储** —— 基于嵌入、用于语义相似度搜索的存储，如 pgvector、Pinecone、Weaviate、FAISS 或 Chroma。
- **键值存储** —— 简单的按键查找存储，如 Redis 或持久化的 KV 数据库，通过已知的键来取数据。
- **文件** —— 扁平文件或对象存储，如文件系统或对象存储中的 JSON、JSONL、Markdown、CSV、Parquet 或日志。

### 每种类型该存什么

- **结构化存储**
  - 用户画像和偏好
  - 项目元数据和任务状态
  - 情景记录，含 user、任务类型、时间戳、结果、所学教训等字段
  - 需要过滤、排序、连接（join）、可审计性或更新的条目

- **向量存储**
  - 用于语义检索的、已嵌入的文档分块（chunk）
  - 用于相似度搜索的、已嵌入的情景或记忆摘要
  - 应按"含义"而非精确键或精确 SQL 过滤来检索的知识
  - 通常本身不作为唯一真相来源；常与结构化存储或文件搭配使用

- **键值存储**
  - 会话状态
  - 缓存的摘要或缓存的检索结果
  - 当查找天然按稳定键（如 `user_id` 或 `session_id`）进行时的小型持久偏好
  - 不需要复杂查询的快速访问记忆

- **文件**
  - 仅追加（append-only）日志
  - JSON 或 JSONL 事件历史
  - Markdown 笔记和人类可编辑的记忆文档
  - 大型离线数据集、导出、快照和归档
  - 便于携带和查看，但在高并发查询与更新方面弱于数据库

### 一条实用的经验法则

- 当记忆有清晰字段、且你需要可靠的过滤或更新时，使用**结构化存储**。
- 当检索应基于语义相似度时，使用**向量存储**。
- 当查找键已知、且速度比查询灵活性更重要时，使用**键值存储**。
- 当你想要简单持久化、日志、可携带性或人类可读记录时，使用**文件**。

实践中，强大的系统往往把它们组合起来：例如，用结构化存储作为真相来源，用向量索引做语义查找，用键值层做缓存，用文件做日志或离线归档。

## 情景记忆

情景记忆存储过往经历：智能体或用户当时想做什么、采取了哪些行动、发生了什么、以及学到了什么。它对于避免重复失败、复用成功的工作流、以及基于以往结果个性化未来行动，尤其有用。

### 情景记忆从哪里来

情景记忆来自过往的交互历史，而不仅仅来自原始的对话文本。它通常是从"智能体或用户想做什么、采取了哪些行动、出现了哪些结果、涌现了哪些教训"中提炼出来的。

常见来源包括：

- 对话历史
- 工具调用与工具结果
- 任务结果
- 错误、重试与纠正
- 工作流轨迹（trace）
- 任务后的摘要或反思

对话历史是一个重要来源，但仅靠它自身往往并不完整。最有用的情景记忆，通常把"说了什么"与"尝试了什么、实际发生了什么"结合起来。

### 通常存在哪里、如何存

情景记忆通常以结构化事件记录的形式存储，有时在其上叠加一层语义索引。

- **结构化数据库** —— 情景最常见的真相来源。
  示例：一张 Postgres 或 SQLite 表，每一行存储 `episode_id`、`user_id`、`task_type`、`timestamp`、`outcome`、`resolution`、`lesson_learned` 等字段。

- **文档或事件日志存储** —— 当你想要仅追加记录或灵活的 JSON schema 时很有用。
  示例：一个 JSONL 文件，每行是一个情景；或一个文档存储，每个情景保存为一个 JSON 对象。

- **向量存储作为二级索引** —— 便于找到语义相似的过往情景，但通常不适合作为唯一真相来源。
  示例：把一段简短的文本摘要（如 "Flight booking failed because passport name mismatched; validate legal name before booking"）嵌入，存进 pgvector 或 Pinecone，并链接回完整的结构化记录。

一种实用模式是：把完整情景保存在结构化存储中，然后可选地再加一层嵌入索引用于语义搜索。

一条典型的情景记忆记录可能长这样：

```json
{
  "episode_id": "ep_1024",
  "user_id": "u_7",
  "session_id": "s_88",
  "task_type": "travel_booking",
  "timestamp": "2026-05-19T09:30:00Z",
  "goal": "Book a flight to Tokyo",
  "actions": [
    "searched flights",
    "selected itinerary",
    "submitted traveler details"
  ],
  "outcome": "failure",
  "error_reason": "passport name mismatch",
  "resolution": "use the exact legal passport name before purchase",
  "lesson_learned": "validate the passport legal name before booking",
  "tags": ["booking", "passport", "failure"]
}
```

### 如何使用情景记忆

当以往结果能够改变当前的计划、工具选择或答案时，就应使用情景记忆。典型用例包括：

- 重复性的工作流
- 用户特定的周期性任务
- 调试与排障
- 以往失败或成功解法很重要的任务
- 需要从过往行动中学习的智能体循环

一般来说，你不会为每一个查询都去检索情景记忆。它在"当前任务与以往发生过的某事相似"时最有用。

### 检索管线

一条好的情景记忆检索管线，通常是**混合式**的，而非纯语义的。

1. **理解当前请求** —— 识别任务类型、实体、用户、约束，以及以往经验是否可能重要。
2. **过滤候选情景** —— 用元数据（如 `user_id`、`task_type`、时效、项目或成功/失败状态）缩小搜索空间。
3. **可选地做语义搜索** —— 用对紧凑情景摘要的嵌入，检索语义相似的情景。
4. **对候选排序** —— 综合语义相似度、元数据匹配、时效和教训的可操作性来排序。
5. **只选最好的少数几个** —— 注入少量相关情景，而不是整份日志。
6. **转成紧凑的提示词文本** —— 只包含对当前决策有帮助的部分，例如失败原因和教训。

换言之，当前用户查询常常驱动检索，但这个查询通常应结合最近的轮次、对话摘要和任务元数据一起来理解。

### 被注入的记忆长什么样

被注入的情景记忆通常应紧凑且以行动为导向。不要把原始 JSON 直接倒进提示词，而应把情景摘要成一条简短的教训或警示。

一个紧凑的、被注入的记忆块示例：

```text
Relevant past episode:
- Previous flight booking failed because the passport name did not match the legal document.
  Lesson: validate the passport legal name before purchase.
```

### Anthropic Messages API 格式示例

在 Anthropic 的 Messages API 中，与查询相关的情景记忆常被捆绑进最后一条用户消息，从而与当前请求保持相邻。

```json
{
  "system": "You are a helpful travel-planning agent. Follow policy, respect saved preferences, and avoid repeating known failures.",
  "tools": [
    {
      "name": "search_flights",
      "description": "Find available flights that satisfy itinerary and policy constraints.",
      "input_schema": {
        "type": "object",
        "properties": {
          "origin": { "type": "string" },
          "destination": { "type": "string" },
          "date": { "type": "string" }
        },
        "required": ["origin", "destination", "date"]
      }
    }
  ],
  "messages": [
    {
      "role": "user",
      "content": "Can you remember that our team uses a shared approval process?"
    },
    {
      "role": "assistant",
      "content": "Yes. I will keep that in mind for future bookings."
    },
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "Relevant past episode:\n- Previous flight booking failed because the passport name did not match the legal document. Lesson: validate the passport legal name before purchase."
        },
        {
          "type": "text",
          "text": "Please book the next trip with the saved constraints in mind."
        }
      ]
    }
  ]
}
```

这个示例展示了通常的模式：结构化的情景存活在提示词之外的持久存储中，但该情景的一段简短的、与查询相关的摘要，被注入到紧挨当前用户消息的位置。

## 程序性记忆

程序性记忆是智能体关于"如何做事"的记忆。它包含可复用的方法、策略、工作流、检查清单和执行模式，而不是用户事实或过往事件。

典型例子包括：

- 以指令文件或提示词包（prompt pack）形式存储的技能
- 标准操作流程（SOP）
- 合规或审批工作流
- 调试操作手册（playbook）
- 工具使用配方（recipe）
- MCP 集成指南或资源访问模式

### 它与众不同之处

程序性记忆不同于其他记忆类型：

- **稳定画像记忆** 存储关于用户、项目或环境的事实
- **情景记忆** 存储以前发生了什么
- **程序性记忆** 存储如何行动

有些程序性记忆是从经验中学到的，但很大一部分是由开发者、团队或运营人员预先编写的。这意味着它往往部分地存在于用户特定的记忆系统之外——尽管在运行时它仍然像记忆一样发挥作用。

### 通常如何存储

程序性记忆通常存储在：

- 提示词模板或系统指令
- 技能文件，例如 Markdown 流程或操作手册
- 策略文档和内部运行手册（runbook）
- 工作流图或任务定义
- MCP 资源模板或集成元数据

与通常紧凑且机器可读的工具 schema 不同，程序性记忆往往起初是自然语言指令，智能体在相关时加载或遵循它。

### 智能体如何得知技能或流程的存在

智能体通常不应在每一轮都收到每个技能或操作手册的完整文本。相反，它们往往被给予一份轻量的能力注册表，说明有哪些可用、以及何时使用。

一条实用的注册表条目可能包含：

- 技能或流程的名称
- 一段简短描述
- 它存放在哪里，例如文件路径、资源 ID 或模板名称
- 何时应使用它的触发条件
- 任何重要的约束、前置条件或归属规则

例如，运行时可能注入：

```text
Available skills:
- incident-triage: use for production debugging involving logs, alerts, and rollback decisions
- sql-migration-review: use before applying schema changes

Available MCP integrations:
- docs-server: use for internal engineering docs
- resource template ticket/{id}: use to load ticket details when a task references a ticket number
```

这告诉模型存在哪些流程和集成，而不强迫它在提示词里携带每个流程的完整内容。

### 检索与执行模式

程序性记忆通常分两个阶段使用：

1. **发现相关流程** —— 识别哪个技能、工作流、策略或 MCP 集成适用于当前任务。
2. **只加载需要的细节** —— 取出或打开相关流程，然后在遵循它的同时，按需使用工具或检索到的知识。

这与检索外部知识类似，只不过被检索到的条目主要不是关于世界的证据，而是关于"如何行动"的指引。

### 与工具和 MCP 的关系

程序性记忆、工具和 MCP 相关，但并不是一回事：

- **工具** 是带 schema 的可调用动作
- **程序性记忆** 告诉智能体如何以及何时使用这些动作
- **MCP** 是一种协议或集成层，可能暴露工具、资源、提示词或模板

换言之，工具是动作，程序性记忆是方法，而 MCP 常常是访问层的一部分。一个强大的智能体通常三者都需要。

## 智能体记忆流程

描述完整记忆循环的一种更清晰方式如下：

1. **用户发送一条消息**
   一个新请求到达。

2. **理解请求**
   系统识别：
   - 任务类型
   - 用户
   - 关键实体与约束
   - 是否需要检索记忆

3. **检索相关记忆与知识**
   系统只为本轮收集最有用的上下文，可能包括：
   - 来自结构化或键值存储的**稳定用户/画像记忆**
   - 来自向量存储、SQL 数据库、文件或文档存储的**相关外部知识**
   - **相关程序性记忆**，如技能、检查清单、策略或可复用工作流
   - **相关情景记忆**，通过元数据过滤、时效和语义相似度从过往经历中检索
   - 当任务依赖已连接的系统或结构化外部上下文时的**可用 MCP 资源或模板**
   - **最近的对话历史**，以及在需要时的旧轮次摘要

4. **组装上下文**
   系统按有意为之的顺序构建提示词，例如：
   - 系统指令
   - 稳定用户/画像记忆
   - 工具定义
   - 能力注册表 / 程序性上下文
   - 旧轮次的摘要
   - 较低优先级的检索/支撑上下文
   - 最近的轮次
   - 本轮最高优先级的检索文档或情景
   - 当前用户消息

5. **LLM 在组装好的上下文上进行推理**
   模型判断自己能否直接回答，还是需要使用工具、技能、MCP 资源或已存储流程等能力。

6. **能力使用循环（如需要）**
   模型可能：
   - 调用工具，如搜索、代码运行器或 API
   - 加载或遵循某个技能、操作手册、检查清单或策略
   - 查看可用的 MCP 服务器、资源或资源模板
   - 取出当前步骤所需的某个 MCP 资源或其他外部文档

   由此产生的工具输出、已加载资源和流程特定指令会被加回上下文，模型可能再次推理或使用更多能力，然后才产出最终答案。

7. **有选择地写入记忆**
   在交互期间或之后，系统决定什么值得保存：
   - 当发生了有意义的任务、失败、成功或教训时，**保存一个情景**
   - 仅对持久的、高置信度的用户事实或偏好，**更新稳定画像记忆**
   - 通过摘要并嵌入有用的文档、结论或轨迹，**存储值得检索的输出**
   - 在需要时，为未来的上下文管理**摘要/压缩历史**

   实践中，系统常常在每一轮或每个有意义的步骤之后，决定什么值得保存。当上下文压力变高，或当某个重要事件（如一次失败或纠正）应立即被保留时，它也可能在一轮的中途就摘要或有选择地写入记忆。这意味着长期记忆写入与短期上下文压缩是相关但不完全相同的：前者主要关乎持久保留，后者主要关乎把当前交互控制在上下文窗口之内。

8. **交付回答**
   最终答案返回给用户。
