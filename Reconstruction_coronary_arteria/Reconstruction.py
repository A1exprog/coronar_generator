import numpy as np
from scipy.optimize import minimize
from functools import partial
from scipy.optimize import dual_annealing
from Reconstruction_coronary_arteria.Transformation import TransformationMatrix
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import splprep, splev

class Reconstruction_methods:
    def __init__(self, curve_pr1, curve_pr2, radius_pr1,radius_pr2,angles1, angles2):
        self.curve_pr1 = curve_pr1
        self.curve_pr2 = curve_pr2
        self.points1 = [self.curve_pr1[0],self.curve_pr1[-1]]
        self.points2 = [self.curve_pr2[0],self.curve_pr2[-1]]  
        self.angles1 = angles1  
        self.angles2 = angles2 
        self.rot_3d_angles = [0,0,0]
        self.trans_2d= [0,0]
        self.trans_3d= [0,0,0]
        self.matched_sequence_a = []
        self.matched_sequence_b = []
        self.curve_3d = []
        self.lam1_list = []
        self.lam2_list = []
        self.tangent_vectors = []
        self.radius_pr1 = radius_pr1
        self.radius_pr2= radius_pr2
        self.norm_vec_p1 = []
        self.norm_vec_p2 = []
        self.curve_3d_spl_int = []
        self.radius_3d_spl_int = []
    @staticmethod   
    def ray_from_projection(lam,angles,point,rot_angles,trans_2d,trans_3d,dist = 600,f = 1200):
        dxx,dyy  = trans_2d 
        dx,dy,dz  = trans_3d 
        rx,ry,rz = rot_angles
        PositionerPrimaryAngle,PositionerSecondaryAngle = angles
        return TransformationMatrix(1200,600,3).translation2d(dxx,dyy).back_projection(lam).tran(PositionerPrimaryAngle,PositionerSecondaryAngle).translation3d(dx,dy,dz).rot_3d(rx,ry,rz).apply(point)

    @staticmethod
    def ray_distance(variables,coef):
        lam1, lam2 = variables
        p1,p2,q1,q2,a1,a2,b1,b2,rx,ry,rz,dxx,dyy,dx,dy,dz = coef 
        R = np.linalg.norm(Reconstruction_methods.ray_from_projection(lam1,[a1,a2],[p1,p2],[0,0,0],[0,0],[0,0,0]) - Reconstruction_methods.ray_from_projection(lam2,[b1,b2],[q1,q2],[rx,ry,rz],[dxx,dyy],[dx,dy,dz]))
        return R

    @staticmethod
    def two_points_ray_residual(point1, point2,angle1,angle2,rot_angles,trans_2d,trans_3d):
        """
        Вычисляет расстояние между двумя лучами, исходящими из проекций выбранных точек.
        :param angles1: Углы для первой проекции.
        :param angles2: Углы для второй проекции.
        :param point1: Точка с первой проекции.
        :param point2: Точка со второй проекции.
        :return: Минимальное расстояние между лучами.
        """
        coefs = np.hstack((point1, point2,angle1,angle2,rot_angles,trans_2d,trans_3d))
        initial_guess = [600, 600]  # Начальное приближение для параметров длин лучей
        result = minimize( Reconstruction_methods.ray_distance, initial_guess, args=coefs)
        lam1, lam2 = result.x

        # Вычисляем точки на лучах
        s1 = Reconstruction_methods.ray_from_projection(lam1, angle1, point1,[0,0,0],[0,0],[0,0,0])
        s2 = Reconstruction_methods.ray_from_projection(lam2, angle2, point2,rot_angles,trans_2d,trans_3d)

        # Возвращаем расстояние между точками на лучах
        return np.linalg.norm(s1 - s2)

    def objective_function(self,initial_guess=[0,0,0,0,0,0,0,0]):
        """
        Целевая функция для оптимизации.

        :param angles2: Углы для второй проекции.
        :return: Суммарное расстояние между всеми парами лучей.
        """
        rot_angles= initial_guess[0:3]
        trans_2d = initial_guess[3:5]
        trans_3d = initial_guess[5:8]
 
        total_residual = 0
        for point1, point2 in zip(self.points1, self.points2):
            residual = Reconstruction_methods.two_points_ray_residual(point1,point2,self.angles1,self.angles2,rot_angles,trans_2d,trans_3d)
            total_residual += residual  # Используем квадрат расстояния для устойчивости

        return total_residual

    """   
    def calibrate(self):

        bounds = [(-0.1, 0.1), (-0.1, 0.1),(-0.1, 0.1),(-2, 2),(-2,2),(-2,2),(-2,2),(-2,2)]
        # Запускаем оптимизацию только для углов второй проекции
        result = dual_annealing(self.objective_function, bounds=bounds, maxfun=100000, no_local_search=False)
        if result.success:
            optimal_angles2 = result.x
            return optimal_angles2
        else:
            raise RuntimeError("Оптимизация не удалась:", result.message)   
    """

    def calibrate1(self):
        initial_guess = [0,0,0,0,0,0,0,0]
        # Запускаем оптимизацию только для углов второй проекции
        result =  minimize(self.objective_function, initial_guess,method='trust-constr', tol=1e-12, options={'xtol': 1e-12, 'gtol': 1e-12, 'maxiter': 10000})
        optimal_par = result.x
        self.rot_3d_angles = optimal_par[0:3]
        self.trans_2d= optimal_par[3:5]
        self.trans_3d= optimal_par[5:8]   
        

            
    def calibrate(self):
        initial_guess = [0,0,0,0,0,0,0,0]
        # Запускаем оптимизацию только для углов второй проекции
        result =  minimize(self.objective_function, initial_guess,method = 'Nelder-Mead',tol=1e-9)
        optimal_par = result.x
        self.rot_3d_angles = optimal_par[0:3]
        self.trans_2d= optimal_par[3:5]
        self.trans_3d= optimal_par[5:8]   
        """
        if result.success:
            optimal_par = result.x
            self.rot_3d_angles = optimal_par[0:3]
            self.trans_2d= optimal_par[3:5]
            self.trans_3d= optimal_par[5:8]          
            #return optimal_angles2
        else:
            raise RuntimeError("Оптимизация не удалась:", result.message)
        """    
 
    def show_epipolar_line(self):
        curve_pr1 =  self.curve_pr1
        curve_pr2 =  self.curve_pr2
        Positioner1PrimaryAngle,Positioner1SecondaryAngle = self.angles1
        Positioner2PrimaryAngle,Positioner2SecondaryAngle = self.angles2
        tx,ty,tz = self.rot_3d_angles
        dxx,dyy = self.trans_2d
        dx,dy,dz = self.trans_3d
        line1p1 = TransformationMatrix(1200,600,3).back_projection(0).tran(Positioner1PrimaryAngle,Positioner1SecondaryAngle).inv_rot_3d(tx,ty,tz).translation3d(-dx,-dy,-dz).inv_tran(Positioner2PrimaryAngle,Positioner2SecondaryAngle).projection().translation2d(-dxx,-dyy).apply(curve_pr1[0])
        line1p2 = TransformationMatrix(1200,600,3).back_projection(1200).tran(Positioner1PrimaryAngle,Positioner1SecondaryAngle).inv_rot_3d(tx,ty,tz).translation3d(-dx,-dy,-dz).inv_tran(Positioner2PrimaryAngle,Positioner2SecondaryAngle).projection().translation2d(-dxx,-dyy).apply(curve_pr1[0])
        line1 = np.vstack((line1p1,line1p2)).transpose()
        
        line2p1 = TransformationMatrix(1200,600,3).back_projection(0).tran(Positioner1PrimaryAngle,Positioner1SecondaryAngle).inv_rot_3d(tx,ty,tz).translation3d(-dx,-dy,dz).inv_tran(Positioner2PrimaryAngle,Positioner2SecondaryAngle).projection().translation2d(-dxx,-dyy).apply(curve_pr1[-1])
        line2p2 =TransformationMatrix(1200,600,3).back_projection(1200).tran(Positioner1PrimaryAngle,Positioner1SecondaryAngle).inv_rot_3d(tx,ty,tz).translation3d(-dx,-dy,dz).inv_tran(Positioner2PrimaryAngle,Positioner2SecondaryAngle).projection().translation2d(-dxx,-dyy).apply(curve_pr1[-1])
        line2 = np.vstack((line2p1,line2p2)).transpose()
        
        # Создание графика
        plt.figure(figsize=(16, 12))  # Установка размера графика
        
    
        plt.plot(line1[0], line1[1], label='Эпиполярная линия 1', color='blue', linestyle='--')
        plt.plot(line2[0], line2[1], label='Эпиполярная линия 2', color='red', linestyle='--')

    
        plt.scatter(curve_pr2[:,0], curve_pr2[:,1], marker='o',s=3, label='Проекция 2 ', color='black')
        plt.xlabel('Ось X')
        plt.ylabel('Ось Y')
        plt.legend()
        plt.xlim(-57,57) 
        plt.ylim(-57,57)  
        # Отображение графика
        plt.grid(True)  # Включение сетки
        plt.show()
    def partial_matching(self):
        """
        Реализация частичного сопоставления двух последовательностей точек.
        
        Возвращает:
        - matched_sequence_a: Подпоследовательность точек из sequence_a.
        - matched_sequence_b: Подпослательность точек из sequence_b.
        """
        cost_function = partial(Reconstruction_methods.two_points_ray_residual, 
                                angle1 = self.angles1,angle2 = self.angles2,
                                rot_angles=self.rot_3d_angles,trans_2d=self.trans_2d,
                                trans_3d=self.trans_3d)

    
        sequence_a = self.curve_pr1
        sequence_b = self.curve_pr2
        m = len(sequence_a)
        n = len(sequence_b)
    
        # Инициализация матрицы стоимостей
        cost_matrix = np.zeros((m + 1, n + 1))
        cost_matrix[:, 0] = np.inf  # Первый столбец — бесконечность (нельзя сопоставить)
        cost_matrix[0, :] = np.inf  # Первая строка — бесконечность (нельзя сопоставить)
        cost_matrix[0, 0] = 0  # Начальная точка — стоимость 0
    
        # Заполнение матрицы стоимостей
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                cost = cost_function(sequence_a[i - 1], sequence_b[j - 1])
                cost_matrix[i, j] = cost + min(
                    cost_matrix[i - 1, j],  # Пропуск точки в sequence_a
                    cost_matrix[i, j - 1],  # Пропуск точки в sequence_b
                    cost_matrix[i - 1, j - 1]  # Сопоставление точек
                )
    
        # Восстановление оптимального пути
        path = []
        i, j = m, n
        while i > 0 or j > 0:
                if cost_matrix[i, j] == cost_matrix[i - 1, j - 1] + cost_function(sequence_a[i - 1], sequence_b[j - 1]):
                    path.append((i - 1, j - 1))  # Сопоставление точек
                    i -= 1
                    j -= 1
                elif cost_matrix[i, j] == cost_matrix[i - 1, j]+cost_function(sequence_a[i - 1], sequence_b[j - 1]):  # Пропуск точки в sequence_a
                    i -= 1
                elif cost_matrix[i, j] == cost_matrix[i, j - 1]+cost_function(sequence_a[i - 1], sequence_b[j - 1]):  # Пропуск точки в sequence_b
                    j -= 1        
            
        path.reverse()  # Оптимальный путь от начала до конца
    
        # Извлечение подпоследовательностей точек
        self.matched_radius_pr1= np.array([self.radius_pr1[i] for i, _ in path[1:]])
        self.matched_radius_pr2 = np.array([self.radius_pr2[j] for _, j in path[1:]])

        self.matched_sequence_a = np.array([sequence_a[i] for i, _ in path[1:]])
        self.matched_sequence_b = np.array([sequence_b[j] for _, j in path[1:]])

    def reconstruct_3d_curve(self):
        """
        Восстанавливает точки трехмерной кривой
        """
        reconstructed_points = []
        for point1,point2 in zip(self.matched_sequence_a,self.matched_sequence_b):
            point_3d,lam = Reconstruction_methods.point_reconstruction(point1, point2,self.angles1,self.angles2,self.rot_3d_angles,self.trans_2d,self.trans_3d)
            self.lam1_list.append(lam[0])
            self.lam2_list.append(lam[1])
            reconstructed_points.append(point_3d)
        self.curve_3d = np.array(reconstructed_points)
        
    def calc_norm_vectors_2d(self):
        points = self.matched_sequence_a
        N = len(points)
        tangents = np.zeros_like(points) 
        for i in range(1, N - 1):
            tangents[i] = points[i + 1] - points[i - 1]
        tangents[0] = points[1] - points[0]
        tangents[-1] = points[-1] - points[-2]
        tangents = tangents/ np.linalg.norm(tangents, axis=1, keepdims=True)
        self.norm_vec_p1 = np.array([-tangents[:,1],tangents[:,0]]).transpose()
        points = self.matched_sequence_b
        N = len(points)
        tangents = np.zeros_like(points) 
        for i in range(1, N - 1):
            tangents[i] = points[i + 1] - points[i - 1]
        tangents[0] = points[1] - points[0]
        tangents[-1] = points[-1] - points[-2]
        tangents = tangents/ np.linalg.norm(tangents, axis=1, keepdims=True)
        self.norm_vec_p2 = np.array([-tangents[:,1],tangents[:,0]]).transpose()

    def calc_tangent_vectors_3d(self):
        points = self.curve_3d
        N = len(points)
        tangents = np.zeros_like(points) 
        for i in range(1, N - 1):
            tangents[i] = points[i + 1] - points[i - 1]
        tangents[0] = points[1] - points[0]
        tangents[-1] = points[-1] - points[-2]
        tangents = tangents/ np.linalg.norm(tangents, axis=1, keepdims=True)
        self.tangent_vectors = tangents
        
    def  reconstruct_3d_rad(self):
        radius_3d= np.zeros(len(self.curve_3d)) 
        curve_2d = self.matched_sequence_a
        radius = self.matched_radius_pr1
        norm = self.norm_vec_p1
        for i in range(len(self.curve_3d)):
            p1 = curve_2d[i]
            p2 = curve_2d[i] + radius[i] *norm[i]
            s1 = Reconstruction_methods.ray_from_projection(self.lam1_list[i], self.angles1, p1,[0,0,0],[0,0],[0,0,0])
            s2 = Reconstruction_methods.ray_from_projection(self.lam1_list[i], self.angles1, p2,[0,0,0],[0,0],[0,0,0])
            dr = s1-s2
            alpha = np.arccos(np.dot(dr,self.tangent_vectors[i])/np.linalg.norm(dr)/np.linalg.norm(self.tangent_vectors[i]))
            radius_3d[i] = np.linalg.norm(dr)*np.sin(alpha)
        self.radius_3d = radius_3d

    
    @staticmethod
    def point_reconstruction(point1, point2,angles1,angles2,rot_angles,trans_2d,trans_3d):
        """
        Вычисляет положение трехмерной точки
        """
        coefs = np.hstack((point1, point2,angles1,angles2,rot_angles,trans_2d,trans_3d))
        initial_guess = [600, 600]  # Начальное приближение для параметров длин лучей
        result = minimize(Reconstruction_methods.ray_distance, initial_guess, args=coefs)
        lam1, lam2 = result.x
        # записываем lam1 lam2
        # Вычисляем точки на лучах
        s1 = Reconstruction_methods.ray_from_projection(lam1, angles1, point1,[0,0,0],[0,0],[0,0,0])
        s2 = Reconstruction_methods.ray_from_projection(lam2, angles2, point2,rot_angles,trans_2d,trans_3d)
        # Возвращаем точку на середине минимального отрезка
        return (s1+s2)/2,[lam1,lam2]


    def plot_curve_3d(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        # Рисуем кривую
        ax.plot(self.curve_3d[:,0], self.curve_3d[:,1],self.curve_3d[:,2], label='Восстановленная пространственная кривая', color='b')
        # Настройка меток осей
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        # Добавляем легенду
        ax.legend()
        ax.view_init(elev=30, azim=30)
        # Показываем график
        plt.show()

    def show_reprojection_curve_3d(self):
        Positioner1PrimaryAngle,Positioner1SecondaryAngle = self.angles1
        Positioner2PrimaryAngle,Positioner2SecondaryAngle = self.angles2
        tx,ty,tz = self.rot_3d_angles
        dxx,dyy = self.trans_2d
        dx,dy,dz = self.trans_3d
        curve_3d = self.curve_3d
        
        curve_3d_pr2 = TransformationMatrix(1200,600,4).inv_rot_3d(tx,ty,tz).translation3d(-dx,-dy,-dz).inv_tran(Positioner2PrimaryAngle,Positioner2SecondaryAngle).projection().translation2d(-dxx,-dyy).apply_to_curve(curve_3d)
        curve_3d_pr1 = TransformationMatrix(1200,600,4).inv_tran(Positioner1PrimaryAngle,Positioner1SecondaryAngle).projection().apply_to_curve(curve_3d)
        curve_pr1 =  self.curve_pr1
        curve_pr2 =  self.curve_pr2

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))  # 1 строка, 2 столбца графиков
        # Отображение первой кривой
        axes[0].plot(curve_pr1[:,0], curve_pr1[:,1], label="Исходная проекция", color='blue')
        axes[0].plot(curve_3d_pr1[:,0],curve_3d_pr1[:,1], label="Проекция восстановленной кривой", color='red')
        axes[0].set_title("Проеция 1")
        axes[0].set_xlabel("x")
        axes[0].set_ylabel("y")
        axes[0].legend()
        axes[0].grid(True)
        axes[0].set_xlim(-50,50)  # Установка пределов по оси x
        axes[0].set_ylim(-50,50)  # Установка пределов по оси y
        
        
        # Отображение второй кривой
        axes[1].plot(curve_pr2[:,0], curve_pr2[:,1], label="Исходная проекция", color='blue')
        axes[1].plot(curve_3d_pr2[:,0],curve_3d_pr2[:,1], label="Проекция восстановленной кривой", color='red')
        axes[1].set_title("Проеция 2")
        axes[1].set_xlabel("x")
        axes[1].set_ylabel("y")
        axes[1].legend()
        axes[1].grid(True)
        axes[1].set_xlim(-50,50)  # Установка пределов по оси x
        axes[1].set_ylim(-50,50)  # Установка пределов по оси y
        plt.tight_layout()
        plt.show()     


    def show_matched(self):
        Positioner1PrimaryAngle,Positioner1SecondaryAngle = self.angles1
        Positioner2PrimaryAngle,Positioner2SecondaryAngle = self.angles2
        tx,ty,tz = self.rot_3d_angles
        dxx,dyy = self.trans_2d
        dx,dy,dz = self.trans_3d
        curve_3d = self.curve_3d
        
        curve_pr1_match =  self.matched_sequence_a
        curve_pr2_match = self.matched_sequence_b
        curve_pr1 =  self.curve_pr1
        curve_pr2 =  self.curve_pr2

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))  # 1 строка, 2 столбца графиков
        # Отображение первой кривой
        axes[0].plot(curve_pr1[:,0], curve_pr1[:,1], label="Исходная проекция", color='blue')
        axes[0].plot(curve_pr1_match[:,0],curve_pr1_match[:,1], label="Проекция восстановленной кривой", color='red')
        axes[0].set_title("Проеция 1")
        axes[0].set_xlabel("x")
        axes[0].set_ylabel("y")
        axes[0].legend()
        axes[0].grid(True)
        axes[0].set_xlim(-50,50)  # Установка пределов по оси x
        axes[0].set_ylim(-50,50)  # Установка пределов по оси y
        
        
        # Отображение второй кривой
        axes[1].plot(curve_pr2[:,0], curve_pr2[:,1], label="Исходная проекция", color='blue')
        axes[1].plot(curve_pr2_match[:,0],curve_pr2_match[:,1], label="Проекция восстановленной кривой", color='red')
        axes[1].set_title("Проеция 2")
        axes[1].set_xlabel("x")
        axes[1].set_ylabel("y")
        axes[1].legend()
        axes[1].grid(True)
        axes[1].set_xlim(-50,50)  # Установка пределов по оси x
        axes[1].set_ylim(-50,50)  # Установка пределов по оси y
        plt.tight_layout()
        plt.show() 

    def curve_3d_spline_inter(self,s=0.1,l=100):
        #s - главдкость
        #l - количество точек
        # Исходные данные 
        curve_3d =self.curve_3d
        x, y, z = curve_3d[:, 0], curve_3d[:, 1], curve_3d[:, 2]
        # Задаем веса для каждой точки
        weights = np.ones_like(x)  # Начальные веса равны 1
        weights[0] = 10  # Увеличиваем вес первой точки
        weights[-1] = 10  # Увеличиваем вес последней точки
        # Создаем параметрическое представление сглаживающего сплайна с учетом весов
        tck, u = splprep([x, y, z], s=0.5, w=weights)  # Параметр s контролирует степень сглаживания
        # Генерируем новые точки на кривой
        u_new = np.linspace(0, 1, l)
        self.curve_3d_spl_int = np.array(splev(u_new, tck)).transpose()

        radius_3d =self.radius_3d
        # Задаем веса для каждой точки
        weights = np.ones_like(x)  # Начальные веса равны 1
        # Создаем параметрическое представление сглаживающего сплайна с учетом весов
        tck, u = splprep(radius_3d, s=0.5, w=weights)  # Параметр s контролирует степень сглаживания
        # Генерируем новые точки на кривой
        u_new = np.linspace(0, 1, l)
        self.radius_3d_spl_int = np.array(splev(u_new, tck)).transpose()


    def curve_3d_spline_inter_show(self,s=0.5):  
        x_new, y_new, z_new = self.curve_3d.transpose()
        x_new, y_new, z_new = self.curve_3d_spl_int
        # Визуализация
        fig = plt.figure(figsize=(20, 18))
        ax = fig.add_subplot(111, projection='3d')
        
        # Рисуем исходные точки
        ax.scatter(x, y, z, label='Исходные точки', color='r', s=5)
        
        # Рисуем аппроксимирующую кривую
        ax.plot(x_new, y_new, z_new, label='Аппроксимирующая кривая', color='b')
        
        # Отмечаем первую и последнюю точки
        ax.scatter(x[0], y[0], z[0], color='g', s=5, label='Первая точка')
        ax.scatter(x[-1], y[-1], z[-1], color='m', s=5, label='Последняя точка')
        
        # Настройка меток осей
        ax.set_xlabel('X', fontsize=12)
        ax.set_ylabel('Y', fontsize=12)
        ax.set_zlabel('Z', fontsize=12)
        
        # Добавляем легенду
        ax.legend(fontsize=10)
        ax.view_init(elev=30, azim=30)
        # Показываем график
        plt.show()
                