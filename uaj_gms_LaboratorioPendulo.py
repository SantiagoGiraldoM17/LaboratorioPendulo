import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


# Constantes
m = 19099.8526  # masa en kg
g = 9.81        # aceleración de la gravedad en m/s^2
L = 0.82982     # longitud al centro de masa en m
I = 36279.07    # momento de inercia en kg*m^2
theta0 = -61.818 * np.pi / 180  # ángulo inicial en radianes

# Ecuación diferencial del péndulo
def pendulum_equation(t, y):
    theta, omega = y
    dtheta_dt = omega
    domega_dt = -(m * g * L / I) * np.sin(theta)
    return [dtheta_dt, domega_dt]

# Condiciones iniciales: ángulo inicial y velocidad angular inicial
y0 = [theta0, 0]

# Intervalo de tiempo para la solución
t_span = (0, 3.2)  # 10 segundos de simulación
t_eval = np.linspace(t_span[0], t_span[1], 300)  # puntos de evaluación

# Resolver la ecuación diferencial
sol = solve_ivp(pendulum_equation, t_span, y0, t_eval=t_eval, method='RK45')

# Calcular la aceleración angular
acceleration = -(m * g * L / I) * np.sin(sol.y[0])

# Convertir el ángulo a grados
theta_deg = np.degrees(sol.y[0])


# Gráfico de Ángulo vs Tiempo
plt.figure(figsize=(10, 6))
plt.subplot(3, 1, 1)  # 3 filas, 1 columna, primer gráfico
plt.plot(sol.t, theta_deg)
plt.xlabel('Tiempo (s)')
plt.ylabel('Ángulo (grados)')
plt.title('Ángulo vs Tiempo')
plt.grid(True)

# Gráfico de Velocidad Angular vs Tiempo
plt.subplot(3, 1, 2)  # segundo gráfico
plt.plot(sol.t, sol.y[1], "r-")
plt.xlabel('Tiempo (s)')
plt.ylabel('Velocidad angular (rad/s)')
plt.title('Velocidad Angular vs Tiempo')
plt.grid(True)

# Gráfico de Aceleración Angular vs Tiempo
plt.subplot(3, 1, 3)  # tercer gráfico
plt.plot(sol.t, acceleration, "g-")
plt.xlabel('Tiempo (s)')
plt.ylabel('Aceleración angular (rad/s²)')
plt.title('Aceleración Angular vs Tiempo')
plt.grid(True)

# Mostrar los gráficos
plt.tight_layout()
plt.savefig("Graficas.png")
plt.show()



# Calcular la aceleración tangencial
acceleration_tangential = L * -(m * g * L / I) * np.sin(sol.y[0])

# Calcular las coordenadas x e y del centro de masa en milímetros
x_mm = 1000 * -L * np.sin(sol.y[0])
y_mm = 1000 * -L * np.cos(sol.y[0])

# Configurar la animación
fig, ax = plt.subplots()
padding = 200  # padding adicional para asegurar visibilidad
range_x = max(abs(np.min(x_mm)), abs(np.max(x_mm))) + padding
range_y = max(abs(np.min(y_mm)), abs(np.max(y_mm))) + padding
ax.set_xlim((-range_x, range_x))
ax.set_ylim((-range_y, range_y))
line, = ax.plot([], [], linestyle='dotted', lw=1)  # Estilo de línea punteado
vector_tan, = ax.plot([], [], 'r-', lw=2, label='Aceleración Tangencial (mm/s²)')
ax.legend()

def init():
    line.set_data([], [])
    vector_tan.set_data([], [])
    return line, vector_tan

def update(frame):
    line.set_data([0, x_mm[frame]], [0, y_mm[frame]])
    tan_direction = np.array([np.cos(sol.y[0][frame]), np.sin(sol.y[0][frame])])
    vector_tan.set_data([x_mm[frame], x_mm[frame] + 100 * acceleration_tangential[frame] * -tan_direction[0]],
                        [y_mm[frame], y_mm[frame] + 100 * acceleration_tangential[frame] * tan_direction[1]])
    return line, vector_tan

ani = FuncAnimation(fig, update, frames=len(sol.t), init_func=init, blit=True, interval=50)
writer = PillowWriter(fps=15)
ani.save('Animacion.gif', writer=writer)
plt.show()