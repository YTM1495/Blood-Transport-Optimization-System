import pandas as pd
from pulp import *
banks = pd.read_csv("./data/bank.csv")
requests = pd.read_csv("./data/hospital_requests.csv")
travel = pd.read_csv("./data/travel_times.csv")

#print(banks)
#print(requests)
#print(travel_time)

blood_type = requests.loc[0, "blood_type"]
available_banks = banks[banks[blood_type] > 0]
#print(available_banks[blood_type])

supply = available_banks[blood_type].values
demand = [requests.loc[0,"units_required"]]

print(supply)
print(demand)

cost_matrix = []
for bank in available_banks["bank_id"]:
    row = travel[(travel["bank_id"] == bank) & (travel["hospital_id"] == requests.loc[1,"hospital_id"])]
    cost_matrix.append(row["travel_time_minutes"].values[0])
print(cost_matrix)

from pulp import *

problem = LpProblem("BloodTransport", LpMinimize)

x = [LpVariable(f"x{i}", lowBound=0) for i in range(len(supply))]

# Objective function
problem += lpSum(cost_matrix[i] * x[i] for i in range(len(supply)))

# Demand constraint
problem += lpSum(x) == demand[0]

# Supply constraints
for i in range(len(supply)):
    problem += x[i] <= supply[i]

problem.solve()

for i, var in enumerate(x):
    print(available_banks.iloc[i]["bank_name"], "->", var.value())