import numpy as np

def GetBalance(demand : list, supply : list):
  sum_demand = sum(demand)
  sum_supply = sum(supply)
  if sum_supply > sum_demand:
    demand.append(sum_supply-sum_demand)
  elif sum_supply < sum_demand:
    supply.append(sum_demand-sum_supply)

def NorthWestAngleMethod(demand : list, supply : list) -> list:
  cnt_demanders = len(demand)
  cnt_suppliers = len(supply)
  remainder_of_demand = demand.copy()
  remainder_of_supply = supply.copy()
  reference_plan = [[0 for i in range(cnt_demanders)] for i in range(cnt_suppliers)]
  cur_row_idx = 0
  cur_col_idx = 0
  while cur_row_idx <= len(supply) - 1 and cur_col_idx <= len(demand) - 1:
    if remainder_of_demand[cur_col_idx] < remainder_of_supply[cur_row_idx]:
      reference_plan[cur_row_idx][cur_col_idx] = remainder_of_demand[cur_col_idx]
      remainder_of_supply[cur_row_idx] -= remainder_of_demand[cur_col_idx]
      remainder_of_demand[cur_col_idx] = 0
      cur_col_idx += 1
    elif remainder_of_demand[cur_col_idx] > remainder_of_supply[cur_row_idx]:
      reference_plan[cur_row_idx][cur_col_idx] = remainder_of_supply[cur_row_idx]
      remainder_of_demand[cur_col_idx] -= remainder_of_supply[cur_row_idx]
      remainder_of_supply[cur_row_idx] = 0
      cur_row_idx += 1
    else:
      reference_plan[cur_row_idx][cur_col_idx] = remainder_of_supply[cur_row_idx]
      remainder_of_supply[cur_row_idx] = 0
      remainder_of_demand[cur_col_idx] = 0
      cur_col_idx += 1
      cur_row_idx += 1
  return reference_plan

demand = list(map(int, input('Введите потребности: ').split()))
supply = list(map(int, input('Введите поставки: ').split()))

GetBalance(demand, supply)
reference_plan = NorthWestAngleMethod(demand, supply)
print(np.matrix(reference_plan))