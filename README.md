# H4+ 的 FCI 计算（完全自写）

从基组到能量，全部代码都在本仓库：高斯积分、ROHF/UHF 参考态、行列式 FCI。
依赖只有 **numpy + scipy**（ARPACK 与特殊函数），不导入任何量子化学软件。

体系：4 个氢核 + 3 个电子（charge = +1，doublet），正方形 D4h，边长 2.0 bohr
（最近邻 H–H = 2.0 bohr ≈ 1.06 Å）。

## 结果（cc-pVTZ，56 个球谐基函数，86 240 个行列式）

| 量 | 能量 / Hartree |
|---|---|
| 核排斥 | 2.707106781187 |
| ROHF | −1.668548936816 |
| UHF（⟨S²⟩ = 0.7701） | −1.676726279259 |
| **FCI 基态（²E，双重简并）** | **−1.736654859761** |
| FCI 简并伙伴 | −1.736654859761 |
| FCI 下一激发态 | −1.465410107142 |

相关能（FCI₀ − ROHF）= **−0.068106 Ha = −1.85 eV**。
与 pyscf 的差：ROHF 4×10⁻¹³、UHF 2×10⁻⁹、FCI 2×10⁻¹² Ha。

## 物理背景

正方形 H⁺₄ 在 D4h 下，4 个 1s 组合成 a1g + b1g + eg。3 个电子填成 a1g² eg¹，
基态是 **²E 二重简并态**——Jahn–Teller 不稳定，也是强关联的典型场景：单参考的
ROHF 只能占据 eg 的一个分量，FCI 才能给出正确的简并结构。FCI 前两个根在
10⁻¹² Ha 内简并，正是这个对称性的体现。

S_z = +1/2 子空间对 3 电子体系只含 S = 1/2 态（S = 3/2 需要 S_z = ±3/2），
因此该扇区内不存在自旋污染。

## 安装与运行

```bash
python -m h4fci --basis cc-pVTZ --threads 20 --nroots 3   # 直接运行
pip install -e .                                          # 可选：装成命令行工具
h4-fci --basis cc-pVDZ                                    # 之后可以这样调用
```

也可用 Makefile：

```bash
make run BASIS=cc-pVTZ     # 跑一次计算，结果写 results/
make test                  # 积分 + SCF 校验（数秒）
make bench                 # 逐阶段计时，写 docs/timing.md
```

`--side` 改正方形边长（bohr），`--json` 保存标量结果，`--cache` 指定积分缓存目录。
首次运行会计算并缓存 AO 积分（cc-pVTZ 约 41 s），之后直接读取。

## 目录结构

```
.
├── README.md
├── pyproject.toml            打包与命令行入口（h4-fci）
├── Makefile                  run / test / bench / clean
├── h4fci/                    核心包
│   ├── __init__.py
│   ├── __main__.py           python -m h4fci
│   ├── cli.py                命令行入口
│   ├── basis.py              基组解析（BSE NWChem 格式）与 AO 构造
│   ├── integrals.py          ★ 高斯积分（McMurchie–Davidson）
│   ├── scf.py                ★ ROHF（Roothaan 有效 Fock）与 UHF
│   ├── fci.py                ★ 行列式 FCI（Slater–Condon + ARPACK）
│   └── data/                 基组数据 + pyscf 参考值
│       ├── H.STO-3G.nwchem
│       ├── H.cc-pVDZ.nwchem
│       ├── H.cc-pVTZ.nwchem
│       └── reference_pyscf.json
├── tests/
│   ├── test_integrals.py     积分/SCF 独立校验（解析式 + 有限差分）
│   └── benchmark.py          逐阶段计时 → docs/timing.md
├── docs/
│   ├── results.md            结果汇总（含 pyscf 对比）
│   └── timing.md             耗时明细（由 benchmark 生成）
├── results/                  各次计算的 JSON
└── cache/                    积分缓存（可删，自动重建）
```

## 模块说明

### 积分（`integrals.py`）
McMurchie–Davidson：Hermite 展开系数 + Boys 函数的导数（把 `dⁿF₀` 写成
`X^{2u−n}(2ρ)^u F_u` 的整数系数表），最后把每个 shell 块收缩到球谐 d。
45 150 个 shell quartet（8 重对称），cc-pVTZ 约 41 s，ERI 张量 79 MB。

### SCF（`scf.py`）
* ROHF 用 Roothaan 有效 Fock：闭壳/开壳/虚轨道分块 `[Fc Fb Fc; Fb Fc Fa; Fc Fa Fc]`。
* UHF 用 α/β 双 Fock；初猜做自旋破缺，否则会停在受限解。
* DIIS：归一化误差矢量 + 外推后对称化，系数发散时退回普通迭代。
* 近简并体系（STO-3G）自动退化到阻尼迭代。

### FCI（`fci.py`）
* Sz = +1/2 扇区，Nα = 2、Nβ = 1；α 串是轨道对 (i<j)，β 串是单轨道，
  CI 向量存成矩阵 `Psi[I,J]`（cc-pVTZ：1540 × 56 = 86 240）。
* Slater–Condon 分解后利用 Nβ = 1：α 空间部分是稠密矩阵乘（BLAS），
  β 相关部分按 α 串整批向量化；一次 σ 在 cc-pVTZ 上约 140 ms（20 线程）。
* 对角化用 ARPACK（自动处理简并根）；另附自写 Davidson 作为备选。

## 与 pyscf 的对比

参考值存在 `h4fci/data/reference_pyscf.json`，**程序不导入 pyscf**。

### 能量（cc-pVTZ，总能量）

| 量 | 本程序 | pyscf | 差 |
|---|---|---|---|
| ROHF | −1.668548936816 | −1.668548936816 | 4×10⁻¹³ |
| UHF | −1.676726279259 | −1.676726280940 | 2×10⁻⁹ |
| ⟨S²⟩ | 0.770134 | 0.770134 | — |
| FCI₀ | −1.736654859761 | −1.736654859759 | 2×10⁻¹² |
| FCI₂ | −1.465410107142 | −1.465410107140 | 2×10⁻¹² |

三个基组的 FCI：

| basis | ndet | 本程序 | pyscf | 差 |
|---|---|---|---|---|
| STO-3G | 24 | −1.627425978564 | −1.627425978564 | 4×10⁻¹⁵ |
| cc-pVDZ | 3 800 | −1.726034706047 | −1.726034706047 | 4×10⁻¹³ |
| cc-pVTZ | 86 240 | −1.736654859761 | −1.736654859759 | 2×10⁻¹² |

### 耗时（cc-pVTZ，20 线程）

| 阶段 | 本程序 | pyscf |
|---|---|---|
| 2e 积分 | 40.8 s | 0.1 s |
| ROHF / UHF | 0.6 / 1.55 s | 0.2 / 0.1 s |
| AO→MO 变换 | 0.23 s | 0.2 s |
| FCI 构建 | 0.46 s | 0.5 s |
| FCI 对角化 | 74.7 s（246 次 σ） | 47 s |
| **合计** | **约 119 s** | **约 48 s** |

差距来自实现语言：pyscf 的积分和 FCI 是 C/高度优化实现，本程序是纯 Python。
精度上两者一致到 10⁻¹²。

## 验证

* `make test`（即 `tests/test_integrals.py`）：
  * s 型积分对解析闭式，p/d 型对解析式的有限差分，最大相对误差 2.7×10⁻⁶
    （受差分步长限制）；
  * H₂/STO-3G 的 RHF 能量 −1.1167143252（参考 −1.1167143251，差 8×10⁻¹¹）。
* `make bench`：逐阶段计时（结果写 `docs/timing.md`）。
* 与 pyscf 的逐元素 / 逐本征值对比见上表。

## 已知注意事项

* ROHF 在简并开壳层上有多个局部极小。cc-pVTZ 上 10 圈收敛到最低解（与 pyscf
  参考一致到 4×10⁻¹³）；STO-3G 因 4 个轨道里 3 个近简并，需阻尼且能量比参考
  高 7×10⁻⁷ Ha。FCI 与轨道选择无关，最终结果不受影响。
* `cache/` 可以随时删除，只是省去重新计算积分的时间。
