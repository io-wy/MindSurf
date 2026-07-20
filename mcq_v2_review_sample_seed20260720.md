# MCQ 人工抽检样本

- 题库：`/home/oscar/mindsurf-workspace/pretrain-infra-completion/configs/evaluation/pretrain_mcq_benchmark_v2.jsonl`
- 题库 SHA-256：`e0d27b9ed9dad3409b8fe14245e989556836b578fe39082476a9f1d1d90a90b2`
- 抽样种子：`20260720`
- 样本量：32

逐题检查三件事：**答案是否正确**、**题干或选项是否有歧义**、**领域归类是否恰当**。

## 1. `calibration_reasoning_hash`  —  calibration_reasoning

```
问题：数据哈希变化通常说明什么？
答案：
```

- A. `模型必然更好`
- B. `GPU 必然空闲`
- C. `许可证自动通过`
- D. `数据身份发生变化` **← 标注答案**

## 2. `calibration_reasoning_oom`  —  calibration_reasoning

```
问题：避免 GPU OOM 最直接需要约束什么？
答案：
```

- A. `用户名长度`
- B. `所有任务峰值显存总和` **← 标注答案**
- C. `提交信息`
- D. `文件扩展名`

## 3. `calibration_reasoning_parent`  —  calibration_reasoning

```
问题：continuation 实验必须记录什么？
答案：
```

- A. `桌面背景`
- B. `父 checkpoint 身份` **← 标注答案**
- C. `SSH 密码`
- D. `浏览器历史`

## 4. `calibration_reasoning_repetition`  —  calibration_reasoning

```
问题：生成文本大量重复通常属于什么信号？
答案：
```

- A. `许可通过信号`
- B. `退化信号` **← 标注答案**
- C. `磁盘健康信号`
- D. `数据身份`

## 5. `calibration_reasoning_seed`  —  calibration_reasoning

```
问题：第二个独立 seed 的主要作用是什么？
答案：
```

- A. `增加词表`
- B. `改变许可证`
- C. `检查结论稳定性` **← 标注答案**
- D. `压缩 checkpoint`

## 6. `code_dict_key`  —  code

```
问题：Python 字典通过什么访问对应值？
答案：
```

- A. `键` **← 标注答案**
- B. `行号`
- C. `像素`
- D. `端口`

## 7. `code_docker_build`  —  code

```
问题：Docker 根据 Dockerfile 构建镜像常用哪个命令？
答案：
```

- A. `docker ps`
- B. `docker build` **← 标注答案**
- C. `docker logs`
- D. `docker stop`

## 8. `code_http_get`  —  code

```
问题：HTTP 中读取资源通常使用哪个方法？
答案：
```

- A. `DELETE`
- B. `GET` **← 标注答案**
- C. `PATCH`
- D. `CONNECT`

## 9. `code_py_loop`  —  code

```
问题：Python 遍历序列常用哪个关键字？
答案：
```

- A. `try`
- B. `for` **← 标注答案**
- C. `raise`
- D. `lambda`

## 10. `code_transaction`  —  code

```
问题：数据库事务的原子性表示什么？
答案：
```

- A. `全部成功或全部回滚` **← 标注答案**
- B. `查询必然更快`
- C. `数据永不删除`
- D. `只允许一个用户`

## 11. `english_accurate`  —  english

```
Question: Which word means correct and precise?
Answer:
```

- A. `vague`
- B. `random`
- C. `broken`
- D. `accurate` **← 标注答案**

## 12. `english_bird`  —  english

```
Question: Which animal normally has feathers?
Answer:
```

- A. `bird` **← 标注答案**
- B. `fish`
- C. `snake`
- D. `whale`

## 13. `english_comparative`  —  english

```
Question: What is the comparative form of small?
Answer:
```

- A. `smallest`
- B. `smallly`
- C. `smaller` **← 标注答案**
- D. `more smallest`

## 14. `english_ocean`  —  english

```
Question: Which is an ocean?
Answer:
```

- A. `Sahara`
- B. `Alps`
- C. `Pacific` **← 标注答案**
- D. `Amazon city`

## 15. `english_past_walk`  —  english

```
Question: What is the regular past tense of walk?
Answer:
```

- A. `walks`
- B. `walked` **← 标注答案**
- C. `walking`
- D. `walker`

## 16. `long_context_recall_08`  —  long_context

```
阅读批次 108 的记录：样品甲在周一入库，样品乙在周二入库，样品丙在周三入库，样品丁在周四入库。
问题：样品甲何时入库？
答案：
```

- A. `周一` **← 标注答案**
- B. `周二`
- C. `周三`
- D. `周四`

## 17. `long_context_recall_14`  —  long_context

```
阅读批次 114 的记录：样品甲在周三入库，样品乙在周四入库，样品丙在周一入库，样品丁在周二入库。
问题：样品丙何时入库？
答案：
```

- A. `周二`
- B. `周三`
- C. `周一` **← 标注答案**
- D. `周四`

## 18. `long_context_recall_16`  —  long_context

```
阅读批次 116 的记录：样品甲在周一入库，样品乙在周二入库，样品丙在周三入库，样品丁在周四入库。
问题：样品甲何时入库？
答案：
```

- A. `周一` **← 标注答案**
- B. `周二`
- C. `周三`
- D. `周四`

## 19. `long_context_recall_17`  —  long_context

```
阅读批次 117 的记录：样品甲在周二入库，样品乙在周三入库，样品丙在周四入库，样品丁在周一入库。
问题：样品乙何时入库？
答案：
```

- A. `周一`
- B. `周三` **← 标注答案**
- C. `周二`
- D. `周四`

## 20. `long_context_recall_23`  —  long_context

```
阅读批次 123 的记录：样品甲在周四入库，样品乙在周一入库，样品丙在周二入库，样品丁在周三入库。
问题：样品丁何时入库？
答案：
```

- A. `周一`
- B. `周二`
- C. `周四`
- D. `周三` **← 标注答案**

## 21. `long_context_recall_24`  —  long_context

```
阅读批次 124 的记录：样品甲在周一入库，样品乙在周二入库，样品丙在周三入库，样品丁在周四入库。
问题：样品甲何时入库？
答案：
```

- A. `周一` **← 标注答案**
- B. `周二`
- C. `周三`
- D. `周四`

## 22. `math_multiply_01`  —  math

```
问题：8 × 4 等于多少？
答案：
```

- A. `36`
- B. `32` **← 标注答案**
- C. `24`
- D. `12`

## 23. `math_multiply_03`  —  math

```
问题：10 × 6 等于多少？
答案：
```

- A. `66`
- B. `50`
- C. `16`
- D. `60` **← 标注答案**

## 24. `math_multiply_07`  —  math

```
问题：14 × 10 等于多少？
答案：
```

- A. `150`
- B. `126`
- C. `24`
- D. `140` **← 标注答案**

## 25. `math_multiply_22`  —  math

```
问题：29 × 7 等于多少？
答案：
```

- A. `210`
- B. `174`
- C. `203` **← 标注答案**
- D. `36`

## 26. `math_multiply_27`  —  math

```
问题：34 × 3 等于多少？
答案：
```

- A. `105`
- B. `68`
- C. `37`
- D. `102` **← 标注答案**

## 27. `math_multiply_30`  —  math

```
问题：37 × 6 等于多少？
答案：
```

- A. `228`
- B. `185`
- C. `222` **← 标注答案**
- D. `43`

## 28. `zh_fact_binary`  —  zh_fact

```
问题：二进制只使用哪两个数字？
答案：
```

- A. `1和2`
- B. `0和2`
- C. `0和1` **← 标注答案**
- D. `2和3`

## 29. `zh_fact_ice_state`  —  zh_fact

```
问题：冰属于水的哪种物态？
答案：
```

- A. `液态`
- B. `气态`
- C. `固态` **← 标注答案**
- D. `等离子态`

## 30. `zh_fact_meter`  —  zh_fact

```
问题：国际单位制中长度的基本单位是什么？
答案：
```

- A. `升`
- B. `米` **← 标注答案**
- C. `克`
- D. `秒`

## 31. `zh_fact_red_planet`  —  zh_fact

```
问题：常被称为红色星球的是哪颗行星？
答案：
```

- A. `火星` **← 标注答案**
- B. `木星`
- C. `水星`
- D. `海王星`

## 32. `zh_fact_yangtze`  —  zh_fact

```
问题：中国最长的河流是哪一条？
答案：
```

- A. `黄河`
- B. `珠江`
- C. `长江` **← 标注答案**
- D. `黑龙江`

