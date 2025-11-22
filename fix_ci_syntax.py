#!/usr/bin/env python3
"""
修复CI配置中的语法错误
"""
import os
import re


def fix_ci_workflows():
    """修复GitHub Actions工作流中的语法错误"""
    workflows_dir = ".github/workflows"

    for filename in os.listdir(workflows_dir):
        if filename.endswith(".yml") or filename.endswith(".yaml"):
            filepath = os.path.join(workflows_dir, filename)

            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()

            original_content = content

            # 修复多行Python命令的语法错误
            # 将单行多命令改为多行格式
            content = re.sub(
                r"python -c \"try: import src; print\('✅ src package imports'\)\\\\nexcept Exception as e: print\(f'ℹ️ src import: {e}'\)\"",
                'python -c "try:\\n  import src\\n  print(\\"✅ src package imports\\")\\nexcept Exception as e:\\n  print(f\\"ℹ️ src import: {e}\\")"',
                content,
            )

            # 修复其他可能的多行命令
            content = re.sub(
                r"python -c \"[^\"]*\\\\n[^\"]*\"",
                lambda match: match.group(0).replace("\\\\n", "\\n"),
                content,
            )

            if content != original_content:
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(content)
                print(f"✅ 修复CI语法: {filepath}")


def main():
    print("🔧 修复CI配置语法错误")
    print("====================")

    if os.path.exists(".github/workflows"):
        fix_ci_workflows()
        print("🎉 CI配置语法修复完成")
    else:
        print("⚠️ .github/workflows 目录不存在")


if __name__ == "__main__":
    main()
