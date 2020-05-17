# Importamos todas las librerias que usamos.
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from moviepy.editor import VideoFileClip
from scipy.integrate import odeint
plt.rcParams['figure.figsize'] = [12, 12]
plt.rcParams.update({'font.size': 18})
from os import environ
environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
import pygame
import shutil
from pygame.locals import *
import time
from ctypes import windll, Structure, c_long, byref
# Fin de la importación.

# Código para que el programa se ejecute en primer plano en Windows.
class RECT(Structure):
    _fields_ = [
    ('left',    c_long),
    ('top',     c_long),
    ('right',   c_long),
    ('bottom',  c_long),
    ]
    def width(self):  return self.right  - self.left
    def height(self): return self.bottom - self.top


def onTop(window):
    SetWindowPos = windll.user32.SetWindowPos
    GetWindowRect = windll.user32.GetWindowRect
    rc = RECT()
    GetWindowRect(window, byref(rc))
    SetWindowPos(window, -1, rc.left, rc.top, 0, 0, 0x0001)

# Fin de el setup del primer plano.

# Aqui preguntamos por cada variable que necesitemos.
columns = shutil.get_terminal_size().columns
spacerAdapted = "#" * columns
print(spacerAdapted)
print("Bienvenido/a al resolvedor numérico de ecuaciones de calor en una dimensión!".center(columns))
print("Versión 1.0".center(columns))
print("Autores: Dario Beltran - Juan Guitarte - Fernando Pascual".center(columns))
print("Contacto: fepaso@edem.es".center(columns))
print(spacerAdapted)

print("")
print("Preferencias:")
print("Ahora debes introducir las variables que quieras aplicar para resolver tu ecuación!")
print("Si quieres usar el valor predeterminado solo tienes que presionar enter.")
longitudBarra = int(input("   -> Introduce la longitud de tu barra (predeterminado = 100): ") or "100")
print(f"      -> {longitudBarra}")
precision = int(input("   -> Introduce la precisión para las condiciones iniciales, por norma general se usa 10 veces la longitud de la barra (predeterminado = 1000): ") or "1000")
print(f"      -> {precision}")
tempMax = int(input("   -> Introduce la temperatura máxima que tendrá tu barra en las condiciones iniciales (predeterminado = 100): ") or "100")
print(f"      -> {tempMax}")
# Fin de las preguntas hasta un poco más tarde.

# Ahora vamos a crear una ventana donde le podamos pedir la función inicial sobre la que aplicar la ecuación del calor.
pygame.init()
puntos = {}
bufferPuntos = []
w = precision
h = int(precision*0.7)
tMax = tempMax
ybase = int(round((h/tMax)*60))
ctey = int(round(h/tMax))

def yproporcional(y):
    yp = int(round(y / ctey))
    return tMax-yp

def ciniciales():
    hot = (226, 50, 1)
    cold = (107, 188, 209)
    go = True

    pygame.mouse.set_cursor(*pygame.cursors.tri_left)

    screen = pygame.display.set_mode((w,h))

    pygame.display.set_caption("Heat EQ. C. Iniciales.")
    screen.fill(cold)

    for x in range(0, w):
        puntos[x] = ybase

    def redraw():
        bufferPuntos.clear()
        for x in puntos:
            bufferPuntos.append((x,puntos[x]))
        screen.fill(cold)
        pygame.draw.lines(screen, hot, False, bufferPuntos, 4)
    redraw()

    clock = pygame.time.Clock()

    onTop(pygame.display.get_wm_info()['window'])

    while go == True:
        clock.tick()
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == K_RETURN:
                    go = False

            if pygame.mouse.get_pressed() == (1,0,0):
                x,y = pygame.mouse.get_pos()
                newy = y

                puntos[x] = newy

                redraw()

            pygame.display.flip()

            if event.type == QUIT:
                go = False


constanteDifusividad = float(input("   -> Introduce la constante de difusividad térmica (predeterminado = 1.5): ") or "1.5")
print(f"      -> {constanteDifusividad}")
print("Ahora se abrirá una ventana, debes dibujar tu función inicial. Cuando hayas terminado, presiona ENTER o CIERRA la ventana para continuar.")
time.sleep(2)
ciniciales()
pygame.quit()
# Fin de la ventana que pregunta por condiciones iniciales y le permite dibujar en ella.

# Preguntamos por la iteración del tiempo.
tiempo = float(input("   -> Introduce la cantidad de tiempo que quieres ver evolucionar tus condiciones iniciales (predeterminado = 20): ") or "20")
print(f"      -> {tiempo}")

# Inicializamos algunas variables obtenidas anteriormente.
cteDifu = constanteDifusividad
L = longitudBarra
N = precision
# El paso necesario dada la longitud y la precisión especificada.
dx = L/N
# Matriz con todos los valores de x posibles dada la precisión y la longitud.
x = np.arange(0, L, dx)

# La función fft.fftfreq nos da las frecuencias de muestreo de la transformada de fourier dadas una Longitud y el paso.
kappa = 2*np.pi*np.fft.fftfreq(N, d=dx)

print("Calculando (puede tardar unos minutos si los cálculos son complejos)...")

# Inicializamos una matriz llena de ceros que llamamos u_inicial del mismo tamaño que la cantidad de puntos de la barra...
# ... que hemos decidido estudiar, especificando la precisión.
u_inicial = np.zeros_like(x)
# Iteramos cada valor que hemos sacado del dibujo de la función inicial
for valorx in puntos:
    # Añadimos cada uno de esos puntos dibujados a la nueva matriz que hemos inicializado.
    u_inicial[int(valorx)] = int(yproporcional(puntos[valorx]))

# Ahora aplicamos una primera transformación de fourier a las condiciones iniciales.
# Usamos la función fft.fft de numpy. Aplica la transformada a una matriz usando el método de Fast Fourier Transform FFT...
# ... ahorrando así tiempo de computación.
u_inicialhat = np.fft.fft(u_inicial)
# Más tarde usaremos u_inicialhat para resolver las ecuaciones diferenciales. La función que las resuelve, odeint, no puede...
# ... trabajar con números imaginarios de forma directa, para poder usarlos debemos guardarlos en otro formato.
# El formato es el siguiente: En lugar de tener una matriz de largo de la longitud de la barra, ahora tendremos una de...
# ... el doble de longitud, en la primera mitad guardaremos la parte real y en la siguiente mitad la imaginaria.
# Este proceso lo podemos hacer con el comando concatenate de numpy.
u_inicialhat_deconstruido = np.concatenate((u_inicialhat.real,u_inicialhat.imag))

# Precisión de tiempo.
dt = 0.1
# Lista de todos los puntos de t, desde 0 hasta la variable tiempo, con paso de la variable dt.
t = np.arange(0, tiempo, dt)

# En la función derivar, calculamos las derivadas de cada punto en un instante t usando el concepto de FFT.
def derivar(uhat_deconstruido, t, kappa, cteDifu):
    # Reconstruimos la matriz a una con parte real e imaginaria en el mismo lugar.
    uhat = uhat_deconstruido[:N] + (1j) * uhat_deconstruido[N:]
    # Aplicamos el concepto de Transformada rápida de fourier FFT.
    d_uhat = -cteDifu**2 * (np.power(kappa,2)) * uhat
    # Deconstruimos la matriz a una con parte real seguida con la parte imaginaria.
    d_uhat_deconstruido = np.concatenate((d_uhat.real,d_uhat.imag)).astype('float64')

    return d_uhat_deconstruido


# Aqui usamos la función odeint de Scipy. Esta procesa la matriz u_inicialhat_deconstruido resolviendo...
# ... el sistema de ecuaciones diferenciales ordinarias.
uhat_deconstruido = odeint(derivar, u_inicialhat_deconstruido, t, args=(kappa, cteDifu))
# Reconstruimos la matriz a una con parte real e imaginaria en el mismo lugar.
uhat = uhat_deconstruido[:, :N] + (1j) * uhat_deconstruido[:, N:]

# Inicializamos una matriz igual a uhat pero llena de ceros.
u = np.zeros_like(uhat)

# Iteramos por cada valor de t.
for k in range(len(t)):
    # Aplicamos la FTT inversa con la función fft.ifft de numpy.
    u[k, :] = np.fft.ifft(uhat[k, :])

# Ahora eliminamos el formato de parte real e imaginaria, nos quedamos con lo real para poder usarlo en el plot.
u = u.real

# Creamos una carpeta output para guardar los resultados
if not os.path.exists("output"):
    os.makedirs("output")
# Fin de la creacion de la carpeta

# Limpiamos la carpeta output de la anterior ejecución.
files = glob.glob('output/*')
for f in files:
    os.remove(f)
# Fin de la limpieza.

# Hacemos plot de los resutados obtenidos.
fig = plt.figure()
plt.style.use('dark_background')
ax = fig.add_subplot(111, projection='3d')
plt.set_cmap('hot')
u_plot = u[0:-1:10, :]
for j in range(u_plot.shape[0]):
    ys = j*np.ones(u_plot.shape[1])
    ax.plot(x, ys, u_plot[j, :], alpha=0.5, color=cm.rainbow(j*20))
    # Guardamos cada iteración en la carpeta output para posterior creación de video.
    plt.savefig(f"output/out{'%03d' % j}.png", bbox_inches='tight')

# Guardamos el resultado como una foto.
plt.savefig('output/out3D.png', bbox_inches='tight')
plt.figure()
# Fin del plot 3D.

# Hacemos un segundo plot de como evoluciona en una mapa en 2D.
plt.imshow(np.flipud(u), aspect=8)
plt.axis('off')
plt.show()
# Fin del plot 2D.
print("Imagen 3D resultado guardada en la carpeta output.")


# Creamos un video a partir de todas las imagenes creadas que se encuentran en la carpeta output.
FPS = 24
os.system(f"ffmpeg -r {FPS} -i output/out%03d.png -vcodec libx264 -y videoSolucion.mp4")
# Fin de la creación del video
print("Video creado y guardado en esta misma carpeta. Reproduciendo...")

# Abrimos el video automáticamente y lo reproducimos 2 veces.
pygame.display.set_caption('Video resultado final')

clip = VideoFileClip('videoSolucion.mp4')
clip.preview()
time.sleep(1)
clip.preview()

pygame.quit()
# Fin.