import csv
from collections import defaultdict

def parse_csv(file_content):
    reader = csv.reader(file_content.split('\n'), delimiter=';')
    next(reader)  # Skip header
    data = [row for row in reader if row]
    return data

def calculate_averages(data):
    mixed_effects_same = defaultdict(list)
    mixed_effects_diff = defaultdict(list)
    fixed_effects_same = defaultdict(list)
    fixed_effects_diff = defaultdict(list)
    flops_only = defaultdict(list)
    mixed_intercept = defaultdict(list)

    for row in data:
        model_type = row[2]
        if row[5] == "NA":
            continue
        rmse = float(row[5]) if row[5] else 0

        if "Flops Only" in model_type and "intercept" not in model_type:
            dependent_var = model_type.split("~")[0].strip()
            flops_only[dependent_var].append(rmse)
        elif "Flops Only + intercept" in model_type:
            dependent_var = model_type.split("~")[0].strip()
            mixed_intercept[dependent_var].append(rmse)
        elif "~ log(FLOPs..1E21.) + (1 | Model.Family)" in model_type:
            dependent_var = model_type.split("~")[0].strip()
            if dependent_var in model_type.split("+")[-1]:
                mixed_effects_same[dependent_var].append(rmse)
            else:
                mixed_effects_diff[dependent_var].append(rmse)
        elif "~ 1 + log(FLOPs..1E21.)" in model_type:
            dependent_var = model_type.split("~")[0].strip()
            if dependent_var in model_type.split("+")[-1]:
                fixed_effects_same[dependent_var].append(rmse)
            else:
                fixed_effects_diff[dependent_var].append(rmse)
    print(fixed_effects_same)
    print(fixed_effects_diff)
    
    # mean
    print("Mean RMSE:")
    print("Flops Only:", sum(sum(v) / len(v) for v in flops_only.values()) / len(flops_only))
    print("Mixed Effects Model (Intercept):", sum(sum(v) / len(v) for v in mixed_intercept.values()) / len(mixed_intercept))
    print("Mixed Effects Model (Same Variable):", sum(sum(v) / len(v) for v in mixed_effects_same.values()) / len(mixed_effects_same))
    print("Mixed Effects Model (Different Variables):", sum(sum(v) / len(v) for v in mixed_effects_diff.values()) / len(mixed_effects_diff))
    print("Fixed Effects Model (Same Variable):", sum(sum(v) / len(v) for v in fixed_effects_same.values()) / len(fixed_effects_same))
    print("Fixed Effects Model (Different Variables):", sum(sum(v) / len(v) for v in fixed_effects_diff.values()) / len(fixed_effects_diff))

    # median
    print("Median RMSE:")
    print("Flops Only:", sorted([sum(v) / len(v) for v in flops_only.values()])[len(flops_only) // 2])
    print("Mixed Effects Model (Intercept):", sorted([sum(v) / len(v) for v in mixed_intercept.values()])[len(mixed_intercept) // 2])
    print("Mixed Effects Model (Same Variable):", sorted([sum(v) / len(v) for v in mixed_effects_same.values()])[len(mixed_effects_same) // 2])
    print("Mixed Effects Model (Different Variables):", sorted([sum(v) / len(v) for v in mixed_effects_diff.values()])[len(mixed_effects_diff) // 2])
    print("Fixed Effects Model (Same Variable):", sorted([sum(v) / len(v) for v in fixed_effects_same.values()])[len(fixed_effects_same) // 2])
    print("Fixed Effects Model (Different Variables):", sorted([sum(v) / len(v) for v in fixed_effects_diff.values()])[len(fixed_effects_diff) // 2])


with open('performance_metrics.csv', 'r') as file:
    csv_content = file.read()
data = parse_csv(csv_content)
calculate_averages(data)