import numpy as np

class TransformationMatrix:
    def __init__(self,f,dist,dim):
        # Инициализируем единичную матрицу (начальное состояние — без преобразований)
        self.matrix = np.eye((dim))  # Используем матрицу 4x4 для работы с пространсвенной  точкой и матрицу 3x3 для работы с двумерной точкой
        self.f = f # фокусное расстояние
        self.dist = dist # расстояние от пациента до приемника
        self.shape = np.shape(self.matrix)


    def tran(self,prim_angle,sec_angle):
        """
        Матрица однородного преобразования.
        Переход к системе координат пациента.
        """
        rotation_matrix_x = np.eye(4)
        rotation_matrix_x[1:3, 1:3] = [
            [np.cos(prim_angle*np.pi/180), -np.sin(prim_angle*np.pi/180)],
            [np.sin(prim_angle*np.pi/180), np.cos(prim_angle*np.pi/180)]
        ]
      
        rotation_matrix_y = np.eye(4)
        rotation_matrix_y[0:3, 0:3] = [
            [np.cos(sec_angle*np.pi/180), 0, np.sin(sec_angle*np.pi/180)],
            [0, 1, 0],
            [-np.sin(sec_angle*np.pi/180), 0, np.cos(sec_angle*np.pi/180)]
        ]
       
        translation_matrix = np.eye(4)
        translation_matrix[:3, 3] = [0, 0, self.dist]
        
        tran_matrix = np.dot(rotation_matrix_y,np.dot( rotation_matrix_x,translation_matrix))
        self.matrix = np.dot(tran_matrix,self.matrix)
        return self

    def inv_tran(self,prim_angle,sec_angle):
        """
        Обратная матрица однородного преобразования.
        Переход к системе координат приемника.
        """
      
        rotation_matrix_x = np.eye(4)
        rotation_matrix_x[1:3, 1:3] = [
            [np.cos(prim_angle*np.pi/180), -np.sin(prim_angle*np.pi/180)],
            [np.sin(prim_angle*np.pi/180), np.cos(prim_angle*np.pi/180)]
        ]
      
        
        rotation_matrix_y = np.eye(4)
        rotation_matrix_y[0:3, 0:3] = [
            [np.cos(sec_angle*np.pi/180), 0, np.sin(sec_angle*np.pi/180)],
            [0, 1, 0],
            [-np.sin(sec_angle*np.pi/180), 0, np.cos(sec_angle*np.pi/180)]
        ]
       
        
        translation_matrix = np.eye(4)
        translation_matrix[:3, 3] = [0, 0, self.dist]
        
        tran_matrix = np.dot(rotation_matrix_y,np.dot( rotation_matrix_x,translation_matrix))
        self.matrix = np.dot(np.linalg.inv(tran_matrix),self.matrix)
        return self
        
    
    def scale(self, sx, sy, sz):
        """
        Масштабирование.
        """
        scale_matrix = np.eye(4)
        scale_matrix[0, 0] = sx
        scale_matrix[1, 1] = sy
        scale_matrix[2, 2] = sz
        self.matrix = np.dot(scale_matrix,self.matrix)
        return self
        
    def projection(self):
        """
        Проекционная матрица.
        Применяется в системе координат камеры.
        Преобразует трехмерные координаты точки в двумерные на плоскости приемника
        """
        projection_matrix = np.array([[self.f,0,0,0],[0,self.f,0,0],[0,0,-1,0]])
        self.matrix = np.dot(projection_matrix,self.matrix)
        return self
        
    def back_projection(self,lam):
        """
        Псевдообратная проекционная матрица.
        Для точки на плоскости приемника строит точки луче в системе координат пациента
        lam - параметр длины. 0 - на плоскости. 1200 - на источнике
        """
        
        back_projection_matrix =np.array([[ lam/self.f,0,0],[0, lam/self.f,0],[0,0,-lam],[0,0,1]])
        self.matrix = np.dot(back_projection_matrix,self.matrix)
        return self
        
    def rot_3d(self,tx,ty,tz):
        rotation_matrix_x = np.eye(4)
        tethx = tx*np.pi/180
        tethy = ty*np.pi/180
        tethz = tz*np.pi/180
        rotation_matrix_x = np.eye(4)
        rotation_matrix_x[1:3, 1:3] = [
            [np.cos(tethx), -np.sin(tethx)],
            [np.sin(tethx), np.cos(tethx)]
        ]
        rotation_matrix_y = np.eye(4)
        rotation_matrix_y[0:3, 0:3] = [
            [np.cos(tethy), 0, np.sin(tethy)],
            [0, 1, 0],
            [-np.sin(tethy), 0, np.cos(tethy)]
        ]

        rotation_matrix_z = np.eye(4)
        rotation_matrix_z[0:3, 0:3] = [
            [np.cos(tethz), -np.sin(tethz), 0],
            [np.sin(tethz), np.cos(tethz), 0],
            [0, 0, 1]
        ]
        rot_matrix = np.dot(rotation_matrix_x,np.dot( rotation_matrix_y,rotation_matrix_z))
        self.matrix = np.dot(rot_matrix,self.matrix)
        return self  

    def inv_rot_3d(self,tx,ty,tz):
        rotation_matrix_x = np.eye(4)
        tethx = tx*np.pi/180
        tethy = ty*np.pi/180
        tethz = tz*np.pi/180
        rotation_matrix_x = np.eye(4)
        rotation_matrix_x[1:3, 1:3] = [
            [np.cos(tethx), -np.sin(tethx)],
            [np.sin(tethx), np.cos(tethx)]
        ]
        rotation_matrix_y = np.eye(4)
        rotation_matrix_y[0:3, 0:3] = [
            [np.cos(tethy), 0, np.sin(tethy)],
            [0, 1, 0],
            [-np.sin(tethy), 0, np.cos(tethy)]
        ]

        rotation_matrix_z = np.eye(4)
        rotation_matrix_z[0:3, 0:3] = [
            [np.cos(tethz), -np.sin(tethz), 0],
            [np.sin(tethz), np.cos(tethz), 0],
            [0, 0, 1]
        ]
        rot_matrix = np.dot(rotation_matrix_x,np.dot( rotation_matrix_y,rotation_matrix_z))
        self.matrix = np.dot(np.linalg.inv(rot_matrix),self.matrix)
        return self
    def translation2d(self,dx,dy):
        translation2d_maxrix = np.array([[1,0,dx],[0,1,dy],[0,0,1]])
        self.matrix = np.dot(translation2d_maxrix,self.matrix)
        return self


    def translation3d(self,dx,dy,dz):
        translation3d_matrix = np.eye(4)
        translation3d_matrix[:3, 3] = [dx, dy, dz]
        self.matrix = np.dot(translation3d_matrix,self.matrix)
        return self
    
    @staticmethod
    def tr_points(points, coef):
        """
        Смещение системы координат изображения:
        1. Смещение по x и y: x' = x - 250, y' = y - 250.
        2. Преобразование координат: x'' = -y', y'' = x'
        3. Масштабирование результатов на коэффициент `coef`
        """
        transformed_points = []
        
        for point in points:
            # Шаг 1: Смещение на -250 по x и y
            x_prime = point[0] - 256
            y_prime = point[1] - 256
            
            # Шаг 2: Преобразование координат
            x_double_prime = -y_prime
            y_double_prime = x_prime
            
            # Шаг 3: Масштабирование
            transformed_points.append((x_double_prime * coef, y_double_prime * coef))
        
        return np.array(transformed_points)

    
    def apply(self, point):
        """
        Применяет текущее преобразование к точке.
        """
        point_homogeneous = np.append(point, 1)  # Переводим в однородные координаты
        transformed_point = np.dot(self.matrix, point_homogeneous)
        return transformed_point[:-1]/transformed_point[-1]  # Возвращаем декартовы координаты

    def apply_to_curve(self, points):
        """
        Применяет текущее преобразование к списку точек.
        """
        # Преобразуем каждую точку в однородные координаты
        points_homogeneous = np.hstack((points, np.ones((len(points), 1))))  # Добавляем столбец из единиц
    
        # Применяем матрицу преобразования ко всем точкам одновременно
        transformed_points_homogeneous = np.dot(points_homogeneous, self.matrix.T)
    
        # Переводим обратно в декартовы координаты
        transformed_points = transformed_points_homogeneous[:, :-1] / transformed_points_homogeneous[:, -1][:, None]
    
        return transformed_points
