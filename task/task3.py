# -*- coding: utf-8 -*-
'''
Моделирование распространения ЭМ волны, падающей на границу
вакуум - идеальный диэлектрик.
Используются граничные условия ABC первой степени.
'''

import numpy
from numpy.fft import fft,fftshift
import tools
import matplotlib.pyplot as plt


class Ricker:
    ''' Класс с уравнением плоской волны для сигнала Вейвлета Рикераа в дискретном виде
    Dr - определяет задержку сигнала.
    Fp - пиковая частота с спектре сигнала.
    Sc - число Куранта.
    eps - относительная диэлектрическая проницаемость среды, в которой расположен источник.
    mu - относительная магнитная проницаемость среды, в которой расположен источник.
    '''

    def __init__(self, Dr, Fp, Sc=1.0, eps=1.0, mu=1.0):
        self.Dr = Dr
        self.Fp = Fp
        self.Sc = Sc
        self.eps = eps
        self.mu = mu

    def getE(self, m, q):
        '''
        Расчет поля E в дискретной точке пространства m
        в дискретный момент времени q
        '''
        return (1-2*(numpy.pi*self.Fp*(q-(m*(self.eps*self.mu)**0.5)/(self.Sc)-self.Dr))**2)/(numpy.exp((numpy.pi*self.Fp*(q-(m*(self.eps*self.mu)**0.5)/(self.Sc)-self.Dr))**2))


if __name__ == '__main__':
    # Волновое сопротивление свободного пространства
    W0 = 120.0 * numpy.pi

    #Скорость света в вакууме
    c=299792458

    #Диэлектрическая проницаемость среды
    Eps=1.5

    #Скорость волны
    v=c/numpy.sqrt(Eps)

    # Размер области моделирования вдоль оси X, в метрах
    X=4.5
    
    # Число Куранта
    Sc = 1.0
    
    # Размер области моделирования в отсчетах
    maxSize = 200

    # Время расчета в отсчетах
    maxTime = 1000

    # Размер дискрета по пространству
    dx=X/maxSize

    #Размер дискрета по времени
    dt=Sc*dx/v

    #Шаг по частоте
    df=1/(maxTime*dt)
    
    # Положение источника в отсчетах
    sourcePos = maxSize//2

    # Датчики для регистрации поля
    probesPos = [160]
    probes = [tools.Probe(pos, maxTime) for pos in probesPos]

    # Параметры среды
    # Диэлектрическая проницаемость
    eps = numpy.ones(maxSize)
    eps[:] = Eps

    # Магнитная проницаемость
    mu = numpy.ones(maxSize - 1)

    Ez = numpy.zeros(maxSize)
    Hy = numpy.zeros(maxSize - 1)
    source = Ricker(30, 0.1, Sc, eps[sourcePos], mu[sourcePos])

    # Ez[1] в предыдущий момент времени
    oldEzLeft = Ez[1]

    # Ez[-2] в предыдущий момент времени
    oldEzRight = Ez[-2]

    # Расчет коэффициентов для граничных условий справа
    
    tempRight = Sc / numpy.sqrt(mu[-1] * eps[-1])
    koeffABCRight = (tempRight - 1) / (tempRight + 1)

    # Параметры отображения поля E
    display_field = Ez
    display_ylabel = 'Ez, В/м'
    display_ymin = -1.1
    display_ymax = 1.1

    # Создание экземпляра класса для отображения
    # распределения поля в пространстве
    display = tools.AnimateFieldDisplay(maxSize,
                                        display_ymin, display_ymax,
                                        display_ylabel)

    display.activate()
    display.drawProbes(probesPos)
    display.drawSources([sourcePos])

    for q in range(maxTime):
        # Расчет компоненты поля H
        Hy = Hy + (Ez[1:] - Ez[:-1]) * Sc / (W0 * mu)
        Hy[0]=0

        # Источник возбуждения с использованием метода
        # Total Field / Scattered Field
        Hy[sourcePos - 1] -= Sc / (W0 * mu[sourcePos - 1]) * source.getE(0, q)

        # Расчет компоненты поля E
        Hy_shift = Hy[: -1]
        Ez[1:-1] = Ez[1: -1] + (Hy[1:] - Hy_shift) * Sc * W0 / eps[1: -1]

        # Источник возбуждения с использованием метода
        # Total Field / Scattered Field
        Ez[sourcePos] += (Sc / (numpy.sqrt(eps[sourcePos] * mu[sourcePos])) *
                          source.getE(-0.5, q + 0.5))

        # Граничные условия ABC первой степени справа и PMC слева
        Ez[0] = 0

        Ez[-1] = oldEzRight + koeffABCRight * (Ez[-2] - Ez[-1])
        oldEzRight = Ez[-2]

        # Регистрация поля в датчиках
        for probe in probes:
            probe.addData(Ez, Hy)

        if q % 2 == 0:
            display.updateData(display_field, q)

    display.stop()

    # Отображение сигнала, сохраненного в датчиках
    tools.showProbeSignals(probes, -1.1, 1.1)
    spectr=fftshift(abs(fft(probe.E)))
    spectr/=numpy.max(spectr)
    freq=numpy.arange(-maxTime/2,maxTime/2)*df
    plt.plot(freq,spectr)
    plt.grid()
    plt.ylabel("$|S(f)/Smax|$")
    plt.xlabel(r"$f$,Гц")
    plt.show()
    
