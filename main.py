from decimal import *
import time
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import numdifftools as nd

getcontext().prec = 10 ** 6

st = time.time()

def f(e):
    return e[0] ** 3 * e[1] ** 2 * (4.0 - e[0] - e[1]) * (-1)

def gradient(x_, y_):
    return nd.Gradient(f)([x_, y_])

x, y = 1, 2
a = 0.004
iter = 500
x_list, y_list, f_list = [], [], []

epsilon = Decimal('1e-15')  # Критерий останова(1)
delta = Decimal('1e-15')  # Критерий останова(2)

for i in range(iter):
    x_list.append(x)
    y_list.append(y)

    f_value = f([x, y])
    f_list.append(f_value)

    grad = gradient(x, y)

    # Критерии останова
    if i > 0 and abs(f_list[-1] - f_list[-2]) < epsilon:
        print("Критерий остановки соблюден..")
        break

    if Decimal(np.linalg.norm([grad[0], grad[1]])) < delta:
        print("Критерий остановки соблюден..")
        break

    x = x - a * grad[0]
    y = y - a * grad[1]

    et = time.time()

elapsed_time = et - st
print('Время исполнения:', elapsed_time, 'секунд')

#Отображение информации о критериях останова

if i == iter - 1:
    print("\nКритерий останова не достигнут.")
else:
    print("\nОкончание на итерации:", i + 1)

# Точное решение
exact_solution = 2, 4/3
# окончательные значения и сравнение с точным решением.
print("\nФинальные значения:")
print("X =", x)
print("Y =", y)
print("Значение финальной функции:", f([x, y]))

# Точное решение
print("\nТочное решение:")
print("X =", exact_solution[0])
print("Y =", exact_solution[1])

# Сравнение с точным решением
tolerance = Decimal('1e-6')  # При необходимости можно изменить отклонения
if abs(x - exact_solution[0]) < tolerance and abs(y - exact_solution[1]) < tolerance:
    print("Окончательные значения близки к точному решению.")
else:
    print("Окончательные значения не близки к точному решению.")
#Число итераций
print("\nЧисло итераций:")
print(iter)

# Инициализируем фигуру
fig = plt.figure(figsize=(14, 12))
ax = fig.add_subplot(111, projection='3d')

# сетка для функциональной поверхности
x_surface = np.linspace(-5, 5, 100)
y_surface = np.linspace(-5, 5, 100)
x_surface, y_surface = np.meshgrid(x_surface, y_surface)
f_surface = x_surface ** 3 * y_surface ** 2 * (4.0 - x_surface - y_surface) * (-1)

# поверхность функции
ax.plot_surface(x_surface, y_surface, f_surface, alpha=0.4, cmap='Blues')

# Построение процесса градиента
ax.plot(x_list, y_list, f_list, '.-', c='red', label='Градиент')

# конечная точка отмечена звездочкой
ax.scatter(x_list[-1], y_list[-1], f_list[-1], c='red', marker='*', s=200, label='наш экстремум')

# метка и заголовок
ax.set_xlabel('x', fontsize=15)
ax.set_ylabel('y', fontsize=15)
ax.set_zlabel('f(x, y)', fontsize=15)
ax.set_title('Градиент', fontsize=20)

# легенда
ax.legend()

# сюжет
plt.show()

