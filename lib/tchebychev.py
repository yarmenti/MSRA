import numpy as np


class TchebychevFun(object):
    def __init__(self, func, inf_borns, sup_borns, dimensions):
        inf_borns = np.array(inf_borns)
        sup_borns = np.array(sup_borns)
        dim = np.array(dimensions)
        
        self.__dim = inf_borns.size
        for x in [sup_borns, dim]:
            if x.size != self.__dim:
                raise Exception("Wrong size for: %s" % x)
                        
        self.__a = np.array(.5 * (sup_borns - inf_borns))
        self.__b = np.array(.5 * (inf_borns + sup_borns))
        
        self.__f = func

        self.__dims = dim
        
        self.__init_nodes()
        self.__init_func_at_nodes()
        self.__init_c()
        
    def __init_nodes(self):
        self.__ii = np.array([np.arange(nb+1) for nb in self.__dims])
        arccos_nodes = [np.pi * (2.*self.__ii[i] + 1.)/(2.*self.__dims[i] + 2.) for i in np.arange(self.__dims.size)]
        self.__nodes = np.array([np.cos(arccos_node) for arccos_node in arccos_nodes])
        
        cos_factors = []
        for ii, dim in zip(self.__ii, self.__dims):
            curr_cos_factor = np.zeros((dim+1, dim+1))
            for i in ii:
                curr_cos_factor[i] = np.cos(i * np.pi * (2.*ii + 1.) / (2.*dim + 2.))
                
            cos_factors.append(curr_cos_factor)
            
        self.__cos_factors = np.array(cos_factors)
            
    def __init_func_at_nodes(self):
        m = np.multiply(self.__a, self.__nodes.T) + self.__b
        m = m.T

        mesh = np.meshgrid(*m)
        mesh_f = [mm.flatten() for mm in mesh]

        self.__fx = [self.__f(*mm) for mm in zip(*mesh_f)]
               
        tmp = [ii.flatten() for ii in np.meshgrid(*self.__ii)]
        self.__x_i = [ii for ii in zip(*tmp)]
        
    def __init_c(self):
        self.__c = np.zeros((self.__dims + 1))
        
        for v_ji in self.__x_i:
            coeff = 1.
            cos_d = dict()
            for ii, (ji, d) in enumerate(zip(v_ji, self.__dims)):
                tmp = 1. if ji == 0 else 2.
                coeff *= tmp / (d + 1.)
                
                cos_d[ii] = self.__cos_factors[ii][ji]
                
            fx_c = 0.
            for v_ki, fx in zip(self.__x_i, self.__fx):
                cos_fact = 1.
                for ii, ki in enumerate(v_ki):
                    cos_fact *= cos_d[ii][ki]
                    
                fx_c += fx * cos_fact
                
            self.__c[v_ji] = coeff * fx_c
        
    def __T(self, v_ji, scaled_x):
        return np.product([np.cos(ji * np.arccos(xi)) for ji, xi in zip(v_ji, scaled_x)])
            
    def __call__(self, *x):
        vx = np.array(x)
        if vx.size != self.__dim:
            raise ValueError("Wrong size of x.")

        scaled_x = np.divide(vx - self.__b, self.__a)
        
        res = 0.
        for v_ji in self.__x_i:
            res += self.__c[v_ji] * self.__T(v_ji, scaled_x)
        
        return res