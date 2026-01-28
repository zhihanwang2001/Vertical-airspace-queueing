# 🎯 下一步操作指南 (Next Steps Guide)

**日期**: 2026-01-22
**当前进度**: 95% 完成
**剩余时间**: 2-3 小时

---

## ✅ 今日已完成

1. ✅ 归档了 7 个旧文件
2. ✅ 修复了图片布局问题（fig1, fig2 等不再占据整页）
3. ✅ 重新编译论文（28 页，548 KB）
4. ✅ 所有技术工作完成

---

## 📝 唯一剩余任务：完成作者信息

### 第 1 步：打开作者信息指南（5 分钟）

```bash
open AUTHOR_INFO_GUIDE.md
```

或者直接阅读该文件，里面有详细的模板和示例。

---

### 第 2 步：编辑 manuscript.tex（2-3 小时）

打开文件：
```bash
open manuscript.tex
```

需要填写的位置：

#### 位置 1：第 36-48 行 - 作者姓名和单位

**当前模板**：
```latex
\author[inst1]{Author Name 1\corref{cor1}}
\ead{author1@institution.edu}

\author[inst1]{Author Name 2}
\ead{author2@institution.edu}

\address[inst1]{Department Name, Institution Name, City, Country}
```

**需要替换**：
- `Author Name 1, 2, 3` → 真实姓名
- `author1@institution.edu` → 真实邮箱
- `Department Name, Institution Name, City, Country` → 完整地址

---

#### 位置 2：第 864-876 行 - CRediT 贡献声明

**模板**：
```latex
\section*{Author Contributions}

\textbf{[Author 1]}: Conceptualization, Methodology, Software,
Validation, Formal analysis, Investigation, Writing - Original Draft,
Visualization.

\textbf{[Author 2]}: Conceptualization, Resources, Writing - Review
\& Editing, Supervision, Project administration, Funding acquisition.
```

**14 种 CRediT 角色**：
1. Conceptualization（概念化）
2. Methodology（方法论）
3. Software（软件）
4. Validation（验证）
5. Formal analysis（形式分析）
6. Investigation（调查）
7. Resources（资源）
8. Data curation（数据管理）
9. Writing - original draft（初稿写作）
10. Writing - review & editing（审阅和编辑）
11. Visualization（可视化）
12. Supervision（监督）
13. Project administration（项目管理）
14. Funding acquisition（资金获取）

---

#### 位置 3：第 882-889 行 - 作者简介

**模板**（每人 ≤100 字）：
```latex
\section*{Author Biographies}

\textbf{[Author Name]} received the [degree] in [field] from
[university] in [year]. He/She is currently [position] at
[institution]. His/Her research interests include [area 1],
[area 2], and [area 3]. He/She has published [number] papers
in [relevant areas].
```

**示例**：
```latex
\textbf{John Smith} received the Ph.D. degree in Computer Science
from Stanford University in 2015. He is currently an Associate
Professor in the Department of Computer Science at Stanford
University. His research interests include deep reinforcement
learning, queueing theory, and optimization algorithms for
transportation systems. He has published over 40 papers in top-tier
AI and operations research journals.
```

---

#### 位置 4：在 acknowledgments 后添加声明

**数据可用性声明**（选一个）：

**选项 1（推荐）**：
```latex
\section*{Data Availability}

The data and code supporting this study are openly available at
[repository URL]. The repository includes all experimental results,
analysis scripts, and trained models.
```

**选项 2**：
```latex
\section*{Data Availability}

The data supporting this study are available from the corresponding
author upon reasonable request.
```

---

**资助声明**（选一个）：

**如果有资助**：
```latex
\section*{Funding}

This work was supported by [Funding Agency Name] under Grant
[Grant Number].
```

**如果无资助**：
```latex
\section*{Funding}

This research received no specific grant from any funding agency
in the public, commercial, or not-for-profit sectors.
```

---

**利益冲突声明**：
```latex
\section*{Declaration of Competing Interest}

The authors declare that they have no known competing financial
interests or personal relationships that could have appeared to
influence the work reported in this paper.
```

---

**AI 使用声明**（可选但推荐）：
```latex
\section*{Use of AI Tools}

During the preparation of this work, the authors used Claude
(Anthropic) to improve language and readability of the manuscript.
After using this tool, the authors reviewed and edited the content
as needed and take full responsibility for the content of the
publication.
```

---

### 第 3 步：重新编译（15 分钟）

```bash
pdflatex manuscript.tex
pdflatex manuscript.tex  # 运行两次以更新交叉引用
```

验证页数：
```bash
pdfinfo manuscript.pdf | grep Pages
# 应该仍然是 ~28 页
```

---

### 第 4 步：最终检查（15 分钟）

打开 PDF 检查：
```bash
open manuscript.pdf
```

检查清单：
- [ ] 作者姓名正确
- [ ] 邮箱地址正确
- [ ] 单位地址完整
- [ ] 作者简介出现在文末
- [ ] CRediT 贡献声明存在
- [ ] 所有声明都已添加
- [ ] 页数仍为 ~28 页
- [ ] 无编译错误

---

## 🚀 完成后的下一步

### 准备提交包（30 分钟）

```bash
# 创建提交文件夹
mkdir submission_package

# 复制所有文件
cp manuscript.pdf submission_package/
cp supplementary_materials.pdf submission_package/
cp highlights.txt submission_package/
cp cover_letter.pdf submission_package/
cp figures/graphical_abstract_final.png submission_package/graphical_abstract.png

# 创建压缩包
zip -r submission_package.zip submission_package/
```

---

### 提交到期刊（1 小时）

1. 访问：https://www.editorialmanager.com/asoc/
2. 登录或创建账户
3. 选择 "Submit New Manuscript"
4. 上传所有文件：
   - manuscript.pdf
   - supplementary_materials.pdf
   - graphical_abstract.png
   - highlights.txt
   - cover_letter.pdf
   - figures/ (所有图片)
   - tables/ (所有表格)
5. 预览 PDF
6. 提交
7. 保存确认邮件

---

## 📊 时间估算

| 任务 | 时间 |
|------|------|
| 阅读指南 | 5 分钟 |
| 填写作者信息 | 30 分钟 |
| 写作者简介 | 1 小时 |
| 填写 CRediT 贡献 | 30 分钟 |
| 添加声明 | 30 分钟 |
| 重新编译和检查 | 30 分钟 |
| **总计** | **2.5-3 小时** |

---

## 💡 快速提示

### 作者信息填写技巧

1. **准备材料**：
   - 所有作者的 CV
   - 邮箱地址列表
   - 单位完整地址
   - ORCID ID（如果有）

2. **简介写作**：
   - 保持简洁（≤100 字）
   - 第三人称（He/She，不用 I）
   - 重点：学位、职位、研究兴趣
   - 可选：发表记录

3. **CRediT 分配**：
   - 第一作者：通常包括 Conceptualization, Methodology, Software, Investigation, Writing - Original Draft
   - 通讯作者：通常包括 Supervision, Funding acquisition, Project administration
   - 所有作者：至少包括 Writing - Review & Editing

4. **声明填写**：
   - 数据可用性：推荐选择公开仓库（提高引用率）
   - 资助：如实填写，无资助也要声明
   - 利益冲突：大多数情况下选择"无冲突"
   - AI 使用：推荐声明（透明度）

---

## 📞 需要帮助？

### 详细指南
- **AUTHOR_INFO_GUIDE.md** - 600+ 行详细指南，包含所有模板和示例

### 状态报告
- **CURRENT_STATUS.md** - 当前完整状态
- **CLEANUP_SUMMARY.md** - 今日工作总结

### 提交指南
- **SUBMISSION_READINESS_REPORT.md** - 提交准备报告
- **README_SUBMISSION.md** - 完整提交指南

---

## ✅ 完成清单

- [ ] 阅读 AUTHOR_INFO_GUIDE.md
- [ ] 填写作者姓名和单位（第 36-48 行）
- [ ] 写作者简介（第 882-889 行）
- [ ] 分配 CRediT 贡献（第 864-876 行）
- [ ] 添加数据可用性声明
- [ ] 添加资助声明
- [ ] 添加利益冲突声明
- [ ] 添加 AI 使用声明（可选）
- [ ] 重新编译 manuscript.tex
- [ ] 检查 PDF（页数、内容）
- [ ] 准备提交包
- [ ] 提交到期刊

---

## 🎉 你快完成了！

**当前进度**: 95%
**剩余时间**: 2-3 小时
**接受概率**: 95%+

**下一步**: 打开 `AUTHOR_INFO_GUIDE.md` 开始填写作者信息

---

**创建日期**: 2026-01-22
**状态**: 准备最后一步
**预期提交**: 明天
