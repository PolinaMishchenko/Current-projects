import numpy as np
import matplotlib.pyplot as plt

from side_functions import inter, get_true_index, courant_number, Envir_data_Reading, Init_prof_Reading, Results_saving

from some_of_GeomSp_Strat_Visc_Relax import GeomSp_Strat_Visc_Relax
from EnvironmentalParameters import EnvironmentalParameters

from wenofv_solver import  WENO_FV_Hopf
from cabaret_solver import CABARET_1D_transfer_Ost_Kha_2018


# --- Настраиваемые поля--------------------------

file_name = 'test_prof9'

delta_tau = 0.001                                # Шаг по профилю 

scheme_nonlin = "Weno"
#scheme_nonlin = "Cabar"

isNonUniform = False
isWind = False

on_GeomSp = True
on_Strat = False
on_Visc = False
on_Relax = False

do_Save = True 

#-------------------------------------------------


del_t = 0.001                                    # шаг по времени для перехода на следующую точку луча (опредляет точность постоения лучей)
                                                 #

phi = 0                                          # азимутальный угол начального отлонения луча R1  (в градусах) 
del_phi = 1                                      # дополнительный азимутальный угол между соседними лучами R1 и R2 (в градусах)  

time_step = 1                                    # Шаг по времени отступа лучей R3 и R4 


M = 600000                                       # начальная длина векторов (с запасом)
Mach = 1.6 
Cruise_altitude = 15760                          # Cruise altitude
StartWrite = Cruise_altitude - 100.584                         
Ground_altitude = 264.069  
StopWrite = Ground_altitude



                # reading from files

root_name = 'C:/Users/misch/OneDrive/Документы/Сверхзвук/Python Program/Files for downloading/' 

z_load_V_E, V0_load, z_load_T, T_load, z_load_Pr, Pr_load, z_load_h, h_load = Envir_data_Reading(root_name)


tol = 0.0*10**(-15)          
epsilon = 0.0*10**(-11)      

z_coef = 0.5                                     # коэф-т запаcа
    
bet_coef = 1.2
gamma = 1.4 
ps_0 = 101325 
T0 = 293.15
R = 8.31

p_ambient, T, c0, ro0, h, V0, R1, AA = EnvironmentalParameters(isNonUniform, isWind, z_load_V_E, V0_load, z_load_T, T_load, z_load_Pr, Pr_load, z_load_h, h_load, Cruise_altitude, Mach, StartWrite, StopWrite, tol)

M_load = len(p_ambient)-1
z_load_R = R1[:,2].T

N, tau, p0 = Init_prof_Reading(file_name, root_name, delta_tau, c0[0], p_ambient[0])



pRemember = 2               
z = np.zeros(pRemember)
x = np.zeros([pRemember, 3])



p = np.zeros([1, N + 1])      
p = p0

pP = np.zeros(N)

x[0,:] = R1[0,:]
z[0] = x[0,2]

calc_c0 = np.zeros(2)
calc_AA = np.zeros(2) 
calc_T = np.zeros(2)
calc_p_ambient = np.zeros(2)
calc_ro0 = np.zeros(2)
calc_h = np.zeros(2)
calc_c0[0] = c0[0]
calc_AA[0] = AA[0]
calc_T[0] = T[0]
calc_p_ambient[0] = p_ambient[0]
calc_ro0[0] = ro0[0]
calc_h[0] = h[0]

calc_c0[1] = calc_c0[0]
calc_T[1] = calc_T[0]
calc_p_ambient[1] = calc_p_ambient[0]
calc_ro0[1] = calc_ro0[0]
calc_h[1] = calc_h[0]

PathLength = 0


while (z[0] - StopWrite) >= tol):
       
    COEF = - bet_coef/calc_ro0[0]/calc_c0[0]**3 

    if scheme_nonlin == "Weno":
        
        multiplier_p = COEF
        multiplier_delta_x = 1
        
    else:
        COEF = - COEF
        multiplier_p = 1
        multiplier_delta_x = 1 / COEF
        
    
    bool1 = np.less_equal(np.fabs(p), epsilon)
    cond_zero = get_true_index(bool1)
    if len(cond_zero) == 0:    
        pass
    else:    
        p[cond_zero] = 0     
    
      
    p = p * multiplier_p

    pP = np.add(p[1:],p[:-1])/2  
    
    delta_x, rk_woutCoef = courant_number(z_coef, delta_tau, pP)
    
    delta_x = delta_x * multiplier_delta_x
    
    z[1] = z[0] - delta_x * 1/Mach
    
    if (z[1] - R1[-1,2]) <= tol:
        
        is_end = True
        
        z[1] = R1[-1,2]        
        delta_x = (z[0] - z[1]) * Mach                      

        rk_woutCoef = delta_x * multiplier_delta_x / delta_tau
  
    
    calc_c0[1] = inter(z_load_R, c0, z[1])       
    calc_AA[1] = inter(z_load_R, AA, z[1]) 
    calc_T[1] = inter(z_load_R, T, z[1])
    calc_p_ambient[1] = inter(z_load_R, p_ambient, z[1]) 
    calc_ro0[1] = inter(z_load_R, ro0, z[1])
    calc_h[1] = inter(z_load_R, h, z[1])
    
    
    ''' Ур23 '''

    p = GeomSp_Strat_Visc_Relax(on_GeomSp, on_Strat, on_Visc, on_Relax, p, calc_c0, calc_AA, calc_T, calc_p_ambient, calc_ro0, calc_h, N, delta_x, delta_tau, gamma, ps_0, T0, R)
#        
    ''' Ур23 ''' 
    
    p = np.asarray(p)
    
    
    pP = np.add(p[1:],p[:-1])/2  
    
    
    ''' Ур1 '''
    
    if scheme_nonlin == "Weno":
    
        p = WENO_FV_Hopf(p, tau, delta_x, delta_tau, N, tol)
    
    else:
    
        p = CABARET_1D_transfer_Ost_Kha_2018(p, pP, rk_woutCoef, N, tol)
          
    ''' Ур1 '''
 
    
    p = p / multiplier_p    
          

    
    PathLength = PathLength + delta_x
    

    z[0] = z[1]

    calc_c0[0] = calc_c0[1]
    calc_AA[0] = calc_AA[1]
    calc_T[0] = calc_T[1]
    calc_p_ambient[0] = calc_p_ambient[1]
    calc_ro0[0] = calc_ro0[1] 
    calc_h[0] = calc_h[1]

    p = np.asarray(p)

    print(z[0])


                # Построение графиков решения

print('Длина пройденного пути', PathLength)


fig = plt.figure()
ax = fig.add_subplot(111, label="1")
ax.grid()

plt.plot(tau, p0, '--', label = f'{int(R1[0,2])} m')
plt.plot(tau, p, label = f'{int(z[0])} m' )  
   
plt.xlabel("t, s")
plt.ylabel("Δp, Pa")  
plt.legend()

                # Запись в файл
if do_Save == True:
    Results_saving(tau, p, delta_tau, file_name, on_GeomSp, on_Strat, on_Visc, on_Relax, scheme_nonlin)