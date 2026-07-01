import os

import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

# =============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_PATH = os.path.join(SCRIPT_DIR, "..", "input", "mpg.csv")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "..", "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def main():
  mpg_clean = pd.read_csv(INPUT_PATH)
  regression_table(mpg_clean)
  city_figure(mpg_clean)
  hwy_figure(mpg_clean)


def regression_table(data):
  reg_cty = sm.OLS.from_formula('displ ~ cty', data).fit(cov_type='cluster', cov_kwds={'groups': data['year']})
  print(reg_cty.summary())

  reg_hwy = sm.OLS.from_formula('displ ~ hwy', data).fit(cov_type='cluster', cov_kwds={'groups': data['year']})
  print(reg_hwy.summary())

  reg_hwy_cty = sm.OLS.from_formula('displ ~ cty + hwy', data).fit(cov_type='cluster', cov_kwds={'groups': data['year']})
  print(reg_hwy_cty.summary())

  latex_table = reg_hwy_cty.summary().as_latex()
  with open(os.path.join(OUTPUT_DIR, 'table_reg.tex'), 'w') as f:
    f.write(latex_table)

  with open(os.path.join(OUTPUT_DIR, 'table_clustered_reg.tex'), 'w') as f:
    f.write(latex_table)

"""

def export_clustered_regression_table(reg):
  lines = [
    "\\begin{center}",
    "\\begin{tabular}{lrrr}",
    "\\toprule",
    "\\textbf{Variable} & \\textbf{Coefficient} & \\textbf{Std. Error} & \\textbf{p-value} \\\",
    "\\midrule",
  ]

  for name, coef in reg.params.items():
    se = reg.bse[name]
    pval = reg.pvalues[name]
    lines.append(f"\\textbf{{{name}}} & {coef:.3f} & {se:.3f} & {pval:.3f} \\\")

  lines.extend([
    "\\bottomrule",
    "\\end{tabular}",
    "\\end{center}",
  ])

  with open(os.path.join(OUTPUT_DIR, "table_clustered_reg.tex"), "w") as f:
    f.write("\n".join(lines) + "\n")

"""


def hwy_figure(data):
  plt.scatter(data['displ'], data['hwy'], c=data['year'])
  plt.xlabel("Engine displacement (L)")
  plt.ylabel("Highway fuel economy (mpg)")
  plt.savefig(os.path.join(OUTPUT_DIR, "figure_hwy.jpg"))


def city_figure(data):
  plt.scatter(data['displ'], data['cty'], c=data['year'])
  plt.xlabel("Engine displacement (L)")
  plt.ylabel("City fuel economy (mpg, log scale)")
  plt.yscale("log")
  plt.savefig(os.path.join(OUTPUT_DIR, "figure_city.jpg"))

# Execute
if __name__ == "__main__":
  main()
