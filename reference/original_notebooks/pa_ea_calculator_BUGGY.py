import re
import pandas as pd
import json

def normalize_program(prog: str) -> str:
    """Chuẩn hóa program để tính Program Accuracy (PA) theo FinQA protocol."""
    if not prog:
        return ""
    prog = re.sub(r'\s+', '', prog.lower())

    # Giao hoán add/multiply
    prog = re.sub(r'add\(([^,]+),([^)]+)\)', lambda m: 'add(' + ','.join(sorted([m.group(1), m.group(2)])) + ')', prog)
    prog = re.sub(r'multiply\(([^,]+),([^)]+)\)', lambda m: 'multiply(' + ','.join(sorted([m.group(1), m.group(2)])) + ')', prog)

    # Giữ nguyên subtract/divide
    prog = re.sub(r'subtract\(([^,]+),([^)]+)\)', r'subtract(\1,\2)', prog)
    prog = re.sub(r'divide\(([^,]+),([^)]+)\)', r'divide(\1,\2)', prog)

    # Chuẩn hóa table_
    prog = re.sub(r'table_(max|min|average|sum)\(([^,]+),none\)', r'table_\1(\2,None)', prog)

    # Renumber #0, #1,...
    refs = []
    def renumber(m):
        ref = m.group(0)
        if ref not in refs:
            refs.append(ref)
        return f"#{refs.index(ref)}"
    prog = re.sub(r'#\d+', renumber, prog)

    return prog

def execute_program(program: str, table: list) -> float | None:
    """Thực thi program và trả về kết quả cuối cùng (dùng để tính Execution Accuracy - EA)."""
    if not program:
        return None

    # Tạo DataFrame từ table (header ở row 0)
    df = pd.DataFrame()
    if table and isinstance(table, list) and len(table) > 1:
        try:
            headers = table[0]
            rows = table[1:]
            if all(isinstance(r, list) for r in rows):
                df = pd.DataFrame(rows, columns=headers)
                for col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
        except:
            pass

    results = []
    # Tách các lệnh bằng dấu phẩy (không nằm trong ngoặc)
    commands = re.split(r',(?![^()]*\))', program.strip())

    def resolve_arg(arg: str):
        arg = arg.strip()
        if arg.startswith('#'):
            try:
                idx = int(arg[1:])
                return results[idx] if 0 <= idx < len(results) else 0.0
            except:
                return 0.0
        try:
            return float(arg.replace(',', ''))
        except:
            return 0.0

    for cmd in commands:
        cmd = cmd.strip().lower()
        if not cmd:
            continue

        # Arithmetic operations
        if cmd.startswith('add('):
            args = re.findall(r'add\(([^,]+),([^)]+)\)', cmd)
            if args:
                results.append(resolve_arg(args[0][0]) + resolve_arg(args[0][1]))
                continue

        if cmd.startswith('subtract('):
            args = re.findall(r'subtract\(([^,]+),([^)]+)\)', cmd)
            if args:
                results.append(resolve_arg(args[0][0]) - resolve_arg(args[0][1]))
                continue

        if cmd.startswith('multiply('):
            args = re.findall(r'multiply\(([^,]+),([^)]+)\)', cmd)
            if args:
                results.append(resolve_arg(args[0][0]) * resolve_arg(args[0][1]))
                continue

        if cmd.startswith('divide('):
            args = re.findall(r'divide\(([^,]+),([^)]+)\)', cmd)
            if args:
                a, b = resolve_arg(args[0][0]), resolve_arg(args[0][1])
                if abs(b) > 1e-10:
                    results.append(a / b)
                continue

        # Table operations
        for func in ['max', 'min', 'average', 'sum']:
            if cmd.startswith(f'table_{func}('):
                match = re.search(rf'table_{func}\((.+?)(?:,none|,None|\)|$)', cmd, re.I)
                if match:
                    raw_col = match.group(1).strip().strip('"').strip("'")
                    matched_col = None
                    for col in df.columns:
                        if raw_col.lower() in str(col).lower() or str(col).lower() in raw_col.lower():
                            matched_col = col
                            break
                    if matched_col is None and len(df.columns) > 0:
                        matched_col = df.columns[0]
                    if matched_col:
                        try:
                            vals = pd.to_numeric(df[matched_col], errors='coerce').dropna()
                            if len(vals) > 0:
                                val = {'max': vals.max(), 'min': vals.min(), 'average': vals.mean(), 'sum': vals.sum()}[func]
                                results.append(float(val))
                        except:
                            pass
                break

    return results[-1] if results else None

# ==================== DEMO INTERACTIVE (chạy thử nhanh) ====================
if __name__ == "__main__":
    print("=== ViNumQA Program Executor Demo ===\n")
    print("Nhập program (các lệnh ngăn cách bởi dấu phẩy, ví dụ: add(1,2), multiply(#0,100))")
    program = input("Program: ").strip()

    print("\nNhập table (dạng JSON list, ví dụ: [['Year', 'Revenue'], ['2020', 100], ['2021', 200]])")
    print("Hoặc để trống nếu không cần table.")
    table_input = input("Table (JSON): ").strip()
    table = json.loads(table_input) if table_input else []

    # Thực thi
    result = execute_program(program.lower(), table)
    normalized = normalize_program(program)

    print("\n--- Kết quả ---")
    print(f"Execution result (EA): {result}")
    print(f"Normalized program (dùng để tính PA): {normalized}")