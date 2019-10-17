# coding:utf-8
from matplotlib import pyplot as plt
import numpy as np
import math
# from datetime import datetime
import os


def calculate_e_theta(psi_deg, height=False, middle=None, beta_plot=False):
    l_1 = 10
    l_2 = 20
    theta = 90
    kappa = 0.3
    beam_sum = 0
    for beta_s in range(90):
        beam_sum += math.cos(math.radians(beta_s))
        # print(math.cos(math.radians(i)/1.5))
    # print(sigma * 2)
    # alpha_rate = 1 / (beam_sum / 2)
    # print('alpha:', alpha_rate)
    beta = np.arange(-89, 90, 1)
    if height:
        Q = np.empty((0, 3), dtype=np.float)
        e_theta = 0
        for beta_s in beta:
            tan_beta = math.tan(math.radians(beta_s))
            x = (20 * tan_beta - 5) / (1 + tan_beta * math.tan(math.radians(psi_deg)))
            y = -1 * math.tan(math.radians(psi_deg)) * x + 20
            # print('x, y:', x, y)
            if x ** 2 + (y - l_2) ** 2 < l_2 ** 2:
                Q = np.append(Q, np.array([[x, y, beta_s]]), axis=0)
                if beta_plot:
                    plt.scatter(x, y)
        if beta_plot:
            plt.scatter(-5, 0, label='$S_1$')
            plt.scatter(5, 0, label='Mic')
            plt.xlim(-20, 20)
            plt.ylim(-5, 40)
            plt.show()
        # print('Q:', len(Q))
        psi_deg = math.radians(psi_deg)
        for j, beta_s in enumerate(Q[:, 2]):
            g_beta_s = g_directivity_speaker(beta_s)
            s_q = np.array([Q[j, 0] + l_1 / 2, Q[j, 1]])
            n = np.array([- 1 * math.sin(psi_deg), -1 * math.cos(psi_deg)])
            s_q = -1 * s_q
            cos_beta_s = (s_q @ n) / (np.sqrt(n[0] ** 2 + n[1] ** 2) * np.sqrt(s_q[0] ** 2 + s_q[1] ** 2))
            # print('cos_beta_s', cos_beta_s)
            if cos_beta_s < 0:
                cos_beta_s = 0.001
            e_theta += kappa * cos_beta_s * g_beta_s
            # print('e_h', e_theta)
        return e_theta
    
    if middle:
        rho = middle
        Q = np.empty((0, 3), dtype=np.float)  # 反射点Q
        e_middle = 0
        for beta_s in beta:
            tan_beta = math.tan(math.radians(beta_s))
            x = (20 * tan_beta - 5) / (1 + tan_beta * math.tan(math.radians(psi_deg)))
            y = -1 * math.tan(math.radians(psi_deg)) * x + 20
            # print('x, y:', x, y)
            if x ** 2 + (y - l_2) ** 2 < l_2 ** 2:
                Q = np.append(Q, np.array([[x, y, beta_s]]), axis=0)
                if beta_plot:
                    plt.scatter(x, y)
        # if beta_plot:
        #     plt.scatter(-5, 0, label='$S_1$')
        #     plt.scatter(5, 0, label='Mic')
        #     plt.xlim(-20, 20)
        #     plt.ylim(-5, 40)
        #     plt.show()
        # print('Use reflection point:', Q.shape)
        psi_deg = math.radians(psi_deg)
        cos_beta_s_list = []
        cos_beta_r_list = []
        for j, beta_s in enumerate(Q[:, 2]):
            g_beta_s = g_directivity_speaker(beta_s)
            s_q = np.array([Q[j, 0] + l_1 / 2, Q[j, 1]])
            n = np.array([- 1 * math.sin(psi_deg), -1 * math.cos(psi_deg)])
            # n = np.array([Q[j, 0] - math.sin(psi_deg), Q[j, 1] - math.cos(psi_deg)])
            cos_beta_s = kappa * (-1 * s_q @ n) / (np.sqrt(n[0] ** 2 + n[1] ** 2) * np.sqrt(s_q[0] ** 2 + s_q[1] ** 2))
            cos_beta_s_list.append(cos_beta_s)
            r = s_q + 2 * ((-1 * s_q) @ n) * n  # 反射の式
            # r = r/np.sqrt(r[0] ** 2 + r[1] ** 2)
            q_m = np.array([l_1 / 2 - Q[j, 0], - Q[j, 1]])
            cos_beta_r = (r @ q_m) / (np.sqrt(r[0] ** 2 + r[1] ** 2) * np.sqrt(q_m[0] ** 2 + q_m[1] ** 2))
            # print('cos_beta_r', cos_beta_r)
            if cos_beta_r < 0:
                cos_beta_r = 0.001
            '''e_beta_sが負になることがある。少し考えなければ、一旦は負になったら0にめちゃくちゃ近づける'''
            cos_beta_r = cos_beta_r ** rho
            cos_beta_r_list.append(cos_beta_r)
            e_beta_s = g_beta_s * cos_beta_r
            # print('e_m:', e_beta_s)
            e_middle += e_beta_s
            if beta_plot:
                # plt.plot([Q[j, 0], n[0]], [Q[j, 1], n[1]])
                # plt.scatter(n[0], n[1])
                plt.plot([Q[j, 0], Q[j, 0] + r[0]], [Q[j, 1], Q[j, 1] + r[1]])
                plt.plot([- 1 * l_1 / 2, - 1 * l_1 / 2 + s_q[0]], [0, s_q[1]])
        if beta_plot:
            plt.scatter(-5, 0, label='$S_1$')
            plt.scatter(5, 0, label='Mic')
            plt.xlim(-20, 20)
            plt.ylim(-5, 40)
            plt.show()
            # e_theta += kappa * e_beta
        plt.plot(Q[:, 2], cos_beta_s_list, label='s')
        plt.plot(Q[:, 2], cos_beta_r_list, label='r')
        plt.legend()
        plt.title(str(math.degrees(psi_deg)))
        plt.show()
        # print('e_m', e_m)
        return e_middle
    
    else:
        # 点と直線の距離
        r = l_2 / math.sqrt((math.tan(math.radians(psi_deg))) ** 2 + 1)
        # ２次方程式の解の公式
        a = 1 + math.tan(math.radians(psi_deg)) ** 2
        b = 2 * -1 * math.tan(math.radians(psi_deg)) * l_2
        c = -1 * (r ** 2 - l_2 ** 2)
        x_1, x_2 = solve_quadratic_equation(a, b, c)
        if abs(x_1 - x_2) < 1:
            y = -1 * math.tan(math.radians(psi_deg)) * x_1 + l_2
            x = x_1
        else:
            print("ERROR : not heavy solution")
            return 0
        
        m = check_point_order(x, -1 * l_1 / 2)
        s = theta - math.degrees(math.atan(y / m))
        print('s:', s)
        print('x, y ', x, y)
        print(math.degrees(math.radians(theta) - math.atan(y / m)))
        e_theta = math.cos((math.radians(theta) - math.atan(y / m)))
        
        return e_theta, x, y


def g_directivity_speaker(beta_s):
    # g_beta_s = math.cos(math.radians(beta_s))
    g_beta_s = math.exp(-1 * (beta_s ** 2) / (2 * 45 ** 2))
    return g_beta_s


def solve_quadratic_equation(a, b, c):
    d = (b ** 2 - 4 * a * c) ** (1 / 2)
    x_1 = (-b + d) / (2 * a)
    x_2 = (-b - d) / (2 * a)
    
    return x_1, x_2


def check_use_point(l_2, x_1, x_2, y_1, y_2):
    r_1 = x_1 ** 2 + (y_1 - l_2) ** 2
    r_2 = x_2 ** 2 + (y_2 - l_2) ** 2
    # print(r_1, r_2)
    if r_1 > l_2 ** 2:
        if r_2 > l_2 ** 2:
            print("ERROR don't have Q point")
            return
        else:
            return x_2, y_2
    else:
        if r_2 > l_2 ** 2:
            return x_1, y_1
        else:
            if x_1 == x_2:
                return x_1, y_1
            else:
                x = [x_1, x_2]
                y = [y_1, y_2]
                return x, y


def check_point_order(x, l):
    if x > l:
        return x - l
    elif x < l:
        return l - x
    else:
        print("ERROR mother is 0")
        return 0


def low_e(ene_0, single=None, multi_plot=False):
    if single:
        print('***********************************')
        print('【' + str(single) + '】')
        e_theta, x, y = calculate_e_theta(single)
        print('e_theta', e_theta)
    else:
        psi_deg_list = np.arange(-40, 41, 10)
        e_list = []
        for psi_deg in psi_deg_list:
            e_theta, x, y = calculate_e_theta(psi_deg)
            if multi_plot:
                z = np.arange(-20, 20)
                plt.plot(z, -1 * math.tan(math.radians(psi_deg)) * z + 20, label='wall')
                plt.scatter(x, y)
                plt.scatter(-5, 0, label='$S_1$')
                plt.scatter(5, 0, label='Mic')
                plt.xlim(-20, 20)
                plt.ylim(-5, 40)
                plt.title(str(psi_deg))
                plt.show()
            ene_l = e_theta + ene_0
            e_list.append(ene_l)
        return e_list


def middle_e(rho, single=None):
    print('Middle freq')
    if single:
        print('***********************************')
        print('【' + str(single) + '】')
        e_theta = calculate_e_theta(single, middle=True)
        print('e_theta', e_theta)
    else:
        psi_deg_list = np.arange(-40, 41, 10)
        e_list = []
        for psi_deg in psi_deg_list:
            e_mid = calculate_e_theta(psi_deg, middle=rho)
            e_list.append(e_mid)
        return e_list


def hight_e(single=None):
    print('Height freq')
    if single:
        print('***********************************')
        print('【' + str(single) + '】')
        e_theta = calculate_e_theta(single, height=True)
        print('e_theta', e_theta)
    else:
        psi_deg_list = np.arange(-40, 41, 10)
        e_list = []
        for psi_deg in psi_deg_list:
            e_theta = calculate_e_theta(psi_deg, height=True)
            ene_h = e_theta
            e_list.append(ene_h)
        return e_list


def my_makedirs(path):
    if not os.path.isdir(path):
        os.makedirs(path)
        

def test_model(psi_deg=15, theta_deg=90, phi_deg=90):
    # パラメータ
    gamma_sm = 0.9
    gamma_dm = 0.1
    gamma_sh = 0.2
    gamma_dh = 0.8
    # l_1 = 10  # ホントはいらない
    # l_2 = 20  # ホントはいらない
    # beta_s_deg = 0    # ホントはいらない(theta_deg = 90のこと)
    rho = 0.5
    alpha = 2
    
    # 必要な数値の計算(このあと11行はいらない。※座標の計算は必要ない。)
    # tan_beta = math.tan(math.radians(beta_s_deg))
    # x = (l_2 * tan_beta - l_1/2) / (1 + tan_beta * math.tan(math.radians(psi_deg)))
    # y = -1 * math.tan(math.radians(psi_deg)) * x + l_2
    # print('x, y:', x, y)
    # if x ** 2 + (y - l_2) ** 2 > l_2 ** 2:
    #     print("ERROR Q is not exit.")
    #     return 0
    # s_q_cal = np.array([x + l_1 / 2, y])
    # q_m_cal = np.array([l_1 / 2 - x, - y])
    # o = np.array([1, 0])
    # phi_deg_cal = math.degrees(math.acos(((- 1 * q_m_cal) @ o) / (np.sqrt(q_m_cal[0] ** 2 + q_m_cal[1] ** 2))))
    
    # ベクトルのみで同様に計算。
    s_q = np.array([math.cos(math.radians(theta_deg)), math.sin(math.radians(theta_deg))])
    q_m = -1 * np.array([math.cos(math.radians(phi_deg)), math.sin(math.radians(phi_deg))])
    psi_rad = math.radians(psi_deg)
    n = np.array([- 1 * math.sin(psi_rad), -1 * math.cos(psi_rad)])
    r = s_q + 2 * ((-1 * s_q) @ n) * n

    # print("s_q", s_q, s_q_cal)
    # print("n", n)
    # print("r", r)
    # print("q_m", q_m, q_m_cal)
    # print("phi_deg_cal", phi_deg_cal)
    
    # 答えの計算
    specular = (r @ q_m) / (np.sqrt(r[0] ** 2 + r[1] ** 2) * np.sqrt(q_m[0] ** 2 + q_m[1] ** 2))
    diffuse = ((-1 * s_q) @ n) / (np.sqrt(n[0] ** 2 + n[1] ** 2) * np.sqrt(s_q[0] ** 2 + s_q[1] ** 2))
    middle = gamma_sm * specular ** alpha + gamma_dm * diffuse * rho
    high = gamma_sh * specular ** alpha + gamma_dh * diffuse * rho
    
    # print("specular", specular, specular**alpha)
    # print("diffuse", diffuse, rho * diffuse, math.degrees(math.acos(diffuse)))
    # print("middle", middle)
    # print("high", high)
    # print("$beta_r$", math.degrees(math.acos(specular)))
    # print("$beta_q$", math.degrees(math.acos(diffuse)))
    # print("**********************************************")
    # print("Difference", middle/high)
    # print("**********************************************")
    
    return middle/high, middle, high
    

def test_model_freq(gamma_d, gamma_s, psi_deg=10, theta_deg=90, phi_deg=90):
    rho = 0.5
    alpha = 2
    s_q = np.array([math.cos(math.radians(theta_deg)), math.sin(math.radians(theta_deg))])
    q_m = -1 * np.array([math.cos(math.radians(phi_deg)), math.sin(math.radians(phi_deg))])
    psi_rad = math.radians(psi_deg)
    n = np.array([- 1 * math.sin(psi_rad), -1 * math.cos(psi_rad)])
    r = s_q + 2 * ((-1 * s_q) @ n) * n
    specular = (r @ q_m) / (np.sqrt(r[0] ** 2 + r[1] ** 2) * np.sqrt(q_m[0] ** 2 + q_m[1] ** 2))
    diffuse = ((-1 * s_q) @ n) / (np.sqrt(n[0] ** 2 + n[1] ** 2) * np.sqrt(s_q[0] ** 2 + s_q[1] ** 2))

    E = gamma_s * specular ** alpha + gamma_d * diffuse * rho
    
    return E


if __name__ == '__main__':
    # # middle_e(0.5)
    # sigma = 0
    # for i in range(90):
    #     sigma += math.cos(math.radians(i))
    # alpha = 1 / sigma
    #
    # e_0 = alpha * math.cos((math.radians(90)))  # 直接音も入れておく。
    # # e_l = low_e(e_0)
    # e_m = middle_e(0.5)
    # e_h = hight_e()
    #
    # freq = np.arange(100, 2000)
    # e = math.e
    # # gamma_l = 1 - 1 / (1 + e ** (-1 * (freq/15 - 35/3)))
    # gamma_h = 0.8 * 1 / (1 + e ** (-1 * (freq / 150 - 25 / 3)))
    # gamma_m = 1 - gamma_h
    #
    # psi = np.arange(-40, 41, 10)
    # model_array = np.zeros((9, len(freq)), dtype=float)
    # plt.figure(figsize=(5, 7))
    # for i, name in enumerate(psi):
    #     print('++++++++++++++++++++++++++++++++++')
    #     print('【' + str(name) + '】')
    #     # print(e_l[i])
    #     print(e_m[i])
    #     print(e_h[i])
    #     E = (gamma_m * e_m[i] + gamma_h * e_h[i])
    #     model_array[i, :] = E
    #     np.zeros((9, len(freq)), dtype=float)
    #     plt.plot(freq, E, label=str(name))
    # # plt.title('model difference about $\psi$')
    # # plt.ylim(0.38, 0.68)
    # plt.xlim(100, 2000)
    # plt.xlabel('freq[Hz]')
    # plt.ylabel('amp')
    # plt.legend()
    # plt.show()
    #
    # array_path = '../_array/19' + datetime.today().strftime("%m%d")
    # my_makedirs(array_path)
    # file_name = '/' + datetime.today().strftime("%H%M%S")
    # np.save(array_path + file_name + '_model_array', model_array)
    # print('Saved')
    
    test_psi_deg_list = np.arange(-50, 50, 1)
    test_theta_deg_list = np.arange(0, 180, 1)
    test_phi_deg_list = np.arange(0, 180, 1)
    freq_list = np.arange(500, 2000, 10)
    e = math.e
    # gamma_l = 1 - 1 / (1 + e ** (-1 * (freq/15 - 35/3)))
    gamma_d = [0.8 * 1 / (1 + e ** (-1 * (freq / 150 - 25 / 3))) for freq in freq_list]
    gamma_s = [1 - k for k in gamma_d]
    # plt.figure()
    # plt.plot(freq_list, gamma_s, label="s")
    # plt.plot(freq_list, gamma_d, label="d")
    # plt.legend()
    # plt.show()
    
    D_s90list = []
    D_s45list = []
    D_sm45list = []
    m_list = []
    h_list = []
    for i in test_psi_deg_list:
        d, middle, high = test_model(psi_deg=i)
        D_s90list.append(d)
        m_list.append(middle)
        h_list.append(high)
        d2, middle2, high2 = test_model(psi_deg=i, theta_deg=45)
        D_s45list.append(d2)
        # d3, middle3, high3 = test_model(psi_deg=i, theta_deg=135)
        # D_sm45list.append(d3)
    plt.figure()
    plt.plot(test_psi_deg_list, D_s45list, label='$\phi$ = 45')
    plt.plot(test_psi_deg_list, D_s90list, label='$\phi$ = 90')
    # plt.plot(test_psi_deg_list, D_sm45list, label='S = 135')
    # plt.title("$psi$")
    plt.ylim(0, 3)
    plt.xlabel("Azimuth [deg]", fontsize=15)
    plt.ylabel("$D(N|f_m, f_h)$", fontsize=15)
    plt.legend(fontsize=15)
    plt.tick_params(labelsize=15)
    plt.show()
    
    plt.figure()
    plt.plot(test_psi_deg_list, m_list, label="$f$=1000")
    plt.plot(test_psi_deg_list, h_list, label="$f$=2000")
    # plt.title("$psi$")
    # plt.ylim(-3, 3)
    plt.xlabel("Azimuth [deg]", fontsize=15)
    plt.ylabel("$E(N)$", fontsize=15)
    plt.tick_params(labelsize=15)
    plt.legend(fontsize=15)
    # plt.show()
    
    # plt.figure()
    # plt.plot(test_psi_deg_list, h_list, label="$f$=2000")
    # # plt.title("$psi$")
    # # plt.ylim(-3, 3)
    # plt.xlabel("Azimuth [deg]", fontsize=15)
    # plt.ylabel("$E(N)$", fontsize=15)
    # plt.tick_params(labelsize=15)
    # plt.show()
    
    e_10list = []
    e_m10list = []
    for j in range(len(freq_list)):
        e_10list.append(test_model_freq(gamma_d[j], gamma_s[j], psi_deg=0))
        e_m10list.append(test_model_freq(gamma_d[j], gamma_s[j], psi_deg=20))
    
    plt.figure()
    plt.plot(freq_list, e_10list, label="$\psi$=0")
    plt.plot(freq_list, e_m10list, label="$\psi$=20")
    # plt.title("$psi$")
    # plt.ylim(-3, 3)
    plt.xlabel("Frequency [Hz]", fontsize=15)
    plt.ylabel("$E(N)$", fontsize=15)
    plt.tick_params(labelsize=15)
    plt.legend(fontsize=15)
    # plt.show()
    # F_list = []
    # for i in test_theta_deg_list:
    #     F_list.append(test_model(theta_deg=i))
    # plt.plot(test_theta_deg_list, F_list)
    # plt.title("$theta$")
    # plt.show()
    # E_list = []
    # for i in test_phi_deg_list:
    #     E_list.append(test_model(phi_deg=i))
    # plt.plot(test_phi_deg_list, E_list)
    # plt.title("$phi$")
    # plt.show()