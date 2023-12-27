import numpy as np
from itertools import combinations
from fractions import Fraction
from method_of_gauss import solve_linear_system_of_equtations
import pandas as pd
from tabulate import tabulate

class DualSimplexMethod:

  def __init__(self, resources, expenses, prices, volumes, write_answer, show_iterations):
    self.solvable = True
    self.number_of_iterations = 0
    self.write_answer = write_answer
    self.show_iterations = show_iterations
    self.resources = resources
    self.expenses = expenses
    self.prices = prices
    self.volumes = volumes
    self.start()
  
  def start(self):
    self.get_extended_form_of_task()
    self.get_dual_task()
    if self.get_connection_basis():
      self.create_simplex_table()
      self.get_optimal_plan()
    else:
      self.solvable = False
      if self.write_answer:
        with open('solve.txt', 'w') as file:
          file.truncate()
          file.write('Задачу решить невозможно')
      print('Задачу решить невозможно')

  def get_extended_form_of_task(self):
    self.quantity_of_artificial_variables = len(resources) + len([volume for volume in volumes if volume > 0])
    self.quantity_of_default_variables = len(volumes)
    self.quantity_of_vectors = self.quantity_of_default_variables + self.quantity_of_artificial_variables + 1
    self.C = np.append(self.prices, [Fraction(0,1) for i in range(self.quantity_of_artificial_variables)])
    self.B = np.append(self.resources, [volume for volume in self.volumes if volume > 0])
    self.A = [list(string) for string in self.expenses]
    for idx,volume in enumerate(volumes):
      if volume > 0:
        new_string = [Fraction(0,1) for i in range(self.quantity_of_default_variables)]
        new_string[idx] = Fraction(1,1)
        self.A.append(new_string)
    for idx in range(self.quantity_of_artificial_variables):
      string_extension = [Fraction(0, 1) for i in range(self.quantity_of_artificial_variables)]
      string_extension[idx] = Fraction(1,1)
      self.A[idx] += (string_extension)
    self.A = np.array([np.array(string) for string in self.A])
    self.size_of_basis = len(self.A)

  def get_dual_task(self):
    self.C_ = self.B
    self.B_ = self.C
    self.A_ = np.transpose(self.A)

  def get_connection_basis(self):
    getting_connection_basis_is_possible = False
    for idx_of_vectors_of_connection_basis in combinations([i + 1 for i in range(self.quantity_of_vectors - 1)], self.size_of_basis):
      A = np.array([self.A_[idx - 1] for idx in idx_of_vectors_of_connection_basis])
      B = np.array([self.B_[idx - 1] for idx in idx_of_vectors_of_connection_basis])
      solve = list()
      idx_of_vectors_of_connection_basis = list(idx_of_vectors_of_connection_basis)
      if all([solve_linear_system_of_equtations(A, B, solve),
              self.check_restrictions(solve, idx_of_vectors_of_connection_basis),
              self.get_pseudo_plan(idx_of_vectors_of_connection_basis),
              self.get_coefficients_of_expansion(idx_of_vectors_of_connection_basis)]
             ):
              self.idx_of_vectors_of_connection_basis = np.array(idx_of_vectors_of_connection_basis)
              getting_connection_basis_is_possible = True
              return getting_connection_basis_is_possible
    return getting_connection_basis_is_possible

  def check_restrictions(self, solve, idx_of_vectors_of_connection_basis):
    result_of_all_expressions_is_less_than_their_restriction = True
    expressions = [self.A_[idx] for idx in range(self.quantity_of_vectors - 1) if idx + 1 not in idx_of_vectors_of_connection_basis]
    restrictions = [self.B_[idx] for idx in range(self.quantity_of_vectors - 1) if idx + 1 not in idx_of_vectors_of_connection_basis]
    for idx, expression in enumerate(expressions):
      restriction = restrictions[idx]
      result_of_expression = sum([a * b for a in expression for b in solve])
      if result_of_expression < restriction:
        result_of_all_expressions_is_less_than_their_restriction = False
        return result_of_all_expressions_is_less_than_their_restriction
    return result_of_all_expressions_is_less_than_their_restriction

  def get_pseudo_plan(self, idx_of_vectors_of_connection_basis):
    getting_pseudo_plan_is_possible = False
    A = np.transpose(np.array([self.A_[idx - 1] for idx in idx_of_vectors_of_connection_basis])).reshape(self.size_of_basis, self.size_of_basis)
    B = self.B
    solve = list()
    if solve_linear_system_of_equtations(A, B, solve):
      self.plan = np.array(solve)
      getting_pseudo_plan_is_possible = True
      return getting_pseudo_plan_is_possible
    return getting_pseudo_plan_is_possible

  def get_coefficients_of_expansion(self, idx_of_vectors_of_connection_basis):
    getting_coefficients_of_expansion_is_possible = True
    coefficients_of_expansion = list()
    A = np.transpose(np.array([self.A_[idx - 1] for idx in idx_of_vectors_of_connection_basis])).reshape(self.size_of_basis, self.size_of_basis)
    for idx in range(self.quantity_of_vectors - 1):
      if idx + 1 in idx_of_vectors_of_connection_basis:
        coefficients = [Fraction(0, 1) for i in range(self.size_of_basis)]
        coefficients[idx_of_vectors_of_connection_basis.index(idx + 1)] = Fraction(1, 1)
        coefficients_of_expansion.append(np.array(coefficients))
      else:
        B = self.A_[idx]
        solve = list()
        if solve_linear_system_of_equtations(A, B, solve):
          coefficients_of_expansion.append(np.array(solve))
        else:
          getting_coefficients_of_expansion_is_possible = False
          return getting_coefficients_of_expansion_is_possible
    self.coefficients_of_expansion = np.array(coefficients_of_expansion)
    return getting_coefficients_of_expansion_is_possible

  def create_simplex_table(self):
    self.matrix_coefficients = np.append(self.plan, self.coefficients_of_expansion).reshape(self.quantity_of_vectors, self.size_of_basis)
    self.C_for_simplex_table = np.append(Fraction(0,1), self.C)
    self.idx_of_vectors_of_basis = self.idx_of_vectors_of_connection_basis
    self.C_of_basis = self.get_C_of_basis()
    self.delta_string = self.fill_delta_string()
    self.idx_guide_string = self.get_guide_string()
    self.teta_string = self.fill_teta_string()
    self.idx_guide_column = self.get_guide_column()
    self.guide_element = self.get_guide_element()

  def get_C_of_basis(self):
    C_of_basis = np.array([self.C_for_simplex_table[idx] for idx in (self.idx_of_vectors_of_basis)])
    return C_of_basis
  
  def fill_delta_string(self):
    delta_string = np.array([np.sum(self.C_of_basis * vector) - self.C_for_simplex_table[idx] for idx,vector in enumerate(self.matrix_coefficients)])
    return delta_string
  
  def get_guide_string(self):
    min_element = min(self.plan)
    idx_guide_string = list(self.plan).index(min_element)
    return idx_guide_string

  def fill_teta_string(self):
    teta_string = np.array([])
    if self.matrix_coefficients[0][self.idx_guide_string] < 0:
      teta_string = np.append(float('inf'), np.array([-(self.delta_string[i] / self.matrix_coefficients[i][self.idx_guide_string]) if (self.matrix_coefficients[i][self.idx_guide_string] < 0) else float('inf') for i in range(1, self.quantity_of_vectors)]))
    else:
      teta_string = np.array([float('inf') for idx in range(len(self.teta_string))])
    return teta_string
  
  def get_guide_column(self):
    min_element = min(self.teta_string)
    idx_guide_column = list(self.teta_string).index(min_element)
    return idx_guide_column
  
  def get_guide_element(self):
    guide_element = self.matrix_coefficients[self.idx_guide_column][self.idx_guide_string]
    return guide_element

  def get_optimal_plan(self):
    with open('solve.txt', 'w') as file:
      file.truncate()
    while not(self.check_optimality_condition()):
      self.idx_of_vectors_of_basis[self.idx_guide_string] = self.idx_guide_column
      self.C_of_basis = self.get_C_of_basis()
      self.simplex_transformation()
    if self.solvable:
      if not(self.show_iterations):
        print('\nОптимальное решение:')
        self.print_iteration()
        if self.write_answer:
          with open('solve.txt', 'a') as file:
            file.write('\nАлгоритм завершён. Последняя итерация соответствует оптимальному решению.')
      else:
        print('\nАлгоритм завершён. Последняя итерация соответствует оптимальному решению.')
        if self.write_answer:
          with open('solve.txt', 'a') as file:
            file.write('\nАлгоритм завершён. Последняя итерация соответствует оптимальному решению.')
    else:
      print('\nЗадачу решить невозможно.\n')
      if self.write_answer:
        with open('solve.txt', 'w') as file:
          file.truncate()
          file.write('Задачу решить невозможно.')

  def check_optimality_condition(self):
    if self.show_iterations:
      self.print_iteration()
    if self.write_answer:
      self.write_iteration()
    if min(self.matrix_coefficients[0]) >= 0:
      return True
    elif (self.teta_string[self.idx_guide_column] == float('inf')):
        self.solvable = False
        return True
    self.number_of_iterations += 1
    return False

  def simplex_transformation(self):
    self.matrix_coefficients_transformation()
    self.plan = self.matrix_coefficients[0]
    self.delta_string = self.fill_delta_string()
    self.idx_guide_string = self.get_guide_string()
    self.teta_string = self.fill_teta_string()
    self.idx_guide_column = self.get_guide_column()
    self.guide_element = self.get_guide_element()

  def matrix_coefficients_transformation(self):
    copy_of_matrix_coefficients = np.copy(self.matrix_coefficients)
    for idx_column in range(len(self.matrix_coefficients)):
      self.matrix_coefficients[idx_column][self.idx_guide_string] /= self.guide_element
      for idx_string in range(self.size_of_basis):
        if idx_string != self.idx_guide_string:
          self.matrix_coefficients[idx_column][idx_string] -= copy_of_matrix_coefficients[self.idx_guide_column][idx_string] * self.matrix_coefficients[idx_column][self.idx_guide_string]
  
  def print_iteration(self):
    print(f'\nИтерация {self.number_of_iterations}')
    table = np.append(np.transpose(self.matrix_coefficients), [self.delta_string, self.teta_string]).reshape(self.size_of_basis + 2, self.quantity_of_vectors)
    df = pd.DataFrame(table, columns = [f'A{idx}' for idx in range(self.quantity_of_vectors)])
    df = df.set_index([pd.Index([f'x{idx}' for idx in self.idx_of_vectors_of_basis] + ['Δ','ϴ'])])
    print(tabulate(df, headers='keys', tablefmt='psql'))
    print('Значение целевой функции:', float(self.delta_string[0]))
  
  def write_iteration(self):
    with open('solve.txt', 'a') as file:
      file.write(f'\nИтерация {self.number_of_iterations}\n')
      table = np.append(np.transpose(self.matrix_coefficients), [self.delta_string, self.teta_string]).reshape(self.size_of_basis + 2, self.quantity_of_vectors)
      df = pd.DataFrame(table, columns = [f'A{idx}' for idx in range(self.quantity_of_vectors)])
      df = df.set_index([pd.Index([f'x{idx}' for idx in self.idx_of_vectors_of_basis] + ['Δ','ϴ'])])
      file.write(tabulate(df, headers='keys', tablefmt='psql'))
      file.write(f'\nЗначение целевой функции:{float(self.delta_string[0])}\n')

if __name__ == "__main__":
  resources = list()
  expenses = list()
  prices = list()
  volumes = list()
  number_of_products = int(input('Введите количество видов изготавливаемой продукции: '))
  number_of_resources = int(input('Введите количество видов используемых ресурсов: '))
  print('Через пробел в указанном порядке введите:')
  idx_of_products = ', '.join([str(i + 1) for i in range(number_of_products)])
  idx_of_resources = ', '.join([str(i + 1) for i in range(number_of_resources)])
  for i in range(number_of_resources):
    string = np.array(list(map(Fraction, input(f'  * Затраты {i + 1} ресурса на изготовление одной единицы продукции {idx_of_products} видов: ').split())))
    expenses.append(string)
  resources = list(map(Fraction, input(f'  * Имеющиеся объёмы каждого из {idx_of_resources} ресурсов: ').split()))
  volumes = list(map(Fraction, input(f'  * Имеющиеся ограничения на максимальный выпуск каждого из {idx_of_products} вида продукта(если ограничения нет, введите "-"): ').replace('-', '-1').split()))
  prices = list(map(Fraction, input(f'  * Цена одной единицы продукции каждого {idx_of_products} вида продукта: ').split()))
  show_iterations = True if input('Показывать промежуточные итерации (введите "Да", если нужно и "Нет" иначе): ') == 'Да' else False
  write_answer = True if input('Записать решение в файл (введите "Да", если нужно и "Нет" иначе): ') == 'Да' else False
  DualSimplexMethod = DualSimplexMethod(resources=np.array(resources), expenses=np.array(expenses), prices=np.array(prices), volumes=np.array(volumes), write_answer=write_answer, show_iterations=show_iterations)