import re


def parse_dependency(dep_str):
    lhs_str, rhs_str = dep_str.split("->")
    lhs_parts = [x.strip() for x in lhs_str.split(",")]
    rhs_parts = [x.strip() for x in rhs_str.split(",")]

    def parse_parts(parts):
        conditions = []
        for part in parts:
            m = re.match(r"(\w+)@([\d\.]+)", part)
            if m:
                attr = m.group(1)
                thr = float(m.group(2))
                conditions.append((attr, thr))
        return conditions

    return {
        'dep_str': dep_str,
        'lhs': parse_parts(lhs_parts),
        'rhs': parse_parts(rhs_parts)
    }

def rfdParsing(RFD_FILE):
    rfds = []
    with open(RFD_FILE, 'r') as f:
        for line in f:
            line = line.strip()
            # skip header or empty lines
            if not line or line.startswith('*'):
                continue
            # extract dependency string before any tab or 'cc:' marker
            dep_str = line.split('\t')[0]
            dep_str = dep_str.split('Key Dependency')[0].strip()
            # now parse
            try:
                parsed = parse_dependency(dep_str)
                parsed['dep_str'] = dep_str
                rfds.append(parsed)
            except Exception as e:
                print(f"Skipping invalid RFD line: '{line}' -> {e}")
    return rfds