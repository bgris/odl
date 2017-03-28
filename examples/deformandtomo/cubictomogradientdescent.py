#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 14:43:23 2017

@author: bgris
"""



import numpy as np
import odl
import copy
import numpy
#%% --- Create template and displacement field (ground truth) and target (data)--- #


# Template space: discretized functions on the rectangle [-1, 1]^2 with
# 100 samples per dimension. Usage of 'linear' interpolation ensures that
# the template gradient is well-defined.

#templ_space = odl.uniform_discr([-1, -1], [1, 1], (100, 100), interp='linear')
#template = odl.phantom.cuboid(templ_space, [-0.5, -0.25], [0.5, 0.25])

templ_space = odl.uniform_discr(min_pt=[-20, -20], max_pt=[20, 20], shape=[200, 200])
template = odl.phantom.shepp_logan(templ_space, modified=True)
# Create a product space for displacement field
disp_field_space = templ_space.tangent_bundle


# Define a displacement field that bends the template a bit towards the
# upper left. We use a list of 2 functions and discretize it using the
# disp_field_space.element() method.
sigma0 = 20
#CP=[0,0 ; 1,0]
#MOM=[0.4,0]
#disp_func = [
#    lambda x: MOM[0]* np.exp(-((x[0]-CP[0]) ** 2 + (x[1]-CP[1])  ** 2) / (2 * sigma ** 2)),
#    lambda x: MOM[1]* np.exp(-((x[0]-CP[0]) ** 2 + (x[1]-CP[1])  ** 2) / (2 * sigma ** 2))]
#    
#CP=np.matrix('-0.5,0.1;0.5,0')
CP0=[[-5,5],[0,0]]#,[-0.5,0.1],[0.5,0],[-0.5,0.1],[0.5,0]]
#MOM=np.matrix('0.4,0;0.1,-0.2')
MOM0=[[0,2],[9,0]]
CP=copy.deepcopy(CP0)
MOM=copy.deepcopy(MOM0)
N0=len(CP)

def VX(x):
    a=0
    for k in range(0,N0):
        a=a+MOM[k][0]* np.exp(-((x[0]-CP[k][0]) ** 2 + (x[1]-CP[k][1])  ** 2) / (2 * sigma0 ** 2))
    return a
    
def VY(x):
    a=0
    for k in range(0,N0):
        a=a+MOM[k][1]* np.exp(-((x[0]-CP[k][0]) ** 2 + (x[1]-CP[k][1])  ** 2) / (2 * sigma0 ** 2))
    return a

disp_func = [VX,VY]
             
disp_field = disp_field_space.element(disp_func)

# Show template and displacement field
#template.show('Template');
#disp_field.show('Displacement field')


# Initialize the deformation operator with fixed template
deform_op = odl.deform.LinDeformFixedTemplForward(template)

# Apply the deformation operator to get the deformed template (ground truth).
deformed_template = deform_op(disp_field)
#deformed_template.show('Deformed template');

#
## Make a circular cone beam geometry with flat detector
## Angles: uniformly spaced, n = 360, min = 0, max = pi + fan angle
#angle_partition = odl.uniform_partition(0, np.pi, 400)
## Detector: uniformly sampled, n = 558, min = -40, max = 40
#detector_partition = odl.uniform_partition(-40, 40, 558)
## Geometry with large fan angle
#geometry = odl.tomo.FanFlatGeometry(
#    angle_partition, detector_partition, src_radius=1, det_radius=100)


angle_partition = odl.uniform_partition(0, np.pi, 180)
# Detector: uniformly sampled, n = 558, min = -30, max = 30
detector_partition = odl.uniform_partition(-20, 20, 128)
geometry = odl.tomo.Parallel2dGeometry(angle_partition, detector_partition)



# Detector: uniformly sampled, n = 400, min = -30, max = 30
#detector_partition = odl.uniform_partition(-30, 30, 400)
#geometry = odl.tomo.Parallel2dGeometry(angle_partition, detector_partition)

# Create the forward operator
#ray_trafo = odl.tomo.RayTransform(reco_space, geometry)


ray_trafo = odl.tomo.RayTransform(templ_space, geometry)
#ray_trafo = odl.tomo.RayTransform(templ_space, geometry, impl='astra_cuda')


# Create projection data by calling the ray transform on the deformed template
data = ray_trafo(deformed_template)
#data.show(title='Projection deformed data (sinogram)');

# Creating a function obj_fun which takes in input an image and returns the l2 difference between its ray transform and data
obj_fun = odl.solvers.L2NormSquared(ray_trafo.range) * (ray_trafo - data)


# Create projection data by calling the ray transform on the deformed template
data_template = ray_trafo(template)

template.show('Template');
deformed_template.show('Deformed template (Ground truth)');

data_template.show(title='Projection template (sinogram)');
data.show(title='Projection deformed data (ground truth, sinogram)');


 

#%% Gradient descent to estimate the vector field with N control points

stepCP=0.001
stepMOM=0.001
Threshold=0.0001**2
NumberIterMax=500
NumberIter=0
NumberMaxLinSearch=10
Lambda=10

#----- Initialisation -----

Error=1
sigma=0.8*sigma0
sigma=10

Xmax=round(40/sigma)
Ymax=round(40/sigma)
Xlin=numpy.linspace(-20,20,Xmax)
Ylin=numpy.linspace(-20,20,Ymax)
CP=[]
MOM=[]
for i in range(0,Xmax):
    for j in range(0,Ymax):
        CP.append([Xlin[i],Ylin[j]])
        MOM.append([0,0])

N=len(CP) 
disp_field_est=disp_field_space.zero()
Energy=obj_fun(deform_op(disp_field_est))
Energy = Lambda*Energy
Energy_data=Energy
Energy_reg=0
for k in range(0,N):
    for j in range(0,N):
        prod= MOM[k][0]* MOM[j][0] +MOM[k][1]* MOM[j][1]
        Energy_reg=Energy_reg+0.5*prod*np.exp(-((CP[j][0]-CP[k][0]) ** 2 + (CP[j][1]-CP[k][1])  ** 2) / (2 * sigma ** 2))

Energy=Energy+Energy_reg
Energy0=Energy

#CP=[[-0.4,0.1],[0.6,-0.1]]
#MOM=[[0,0],[0,0]]
deformed_template_est=template
NumberDepThre=0
print("Iteration= 0 ; Energy= {} ;  Data attachment term = {}".format(Energy, Energy_data))

#----- Gradient descent -----

while (NumberDepThre<3 and NumberIter<NumberIterMax):
    
#    def VX_verif(x):
#        a=0
#        for k in range(0,N):
#            a=a+MOM[k][0]* np.exp(-((x[0]-CP[k][0]) ** 2 + (x[1]-CP[k][1])  ** 2) / (2 * sigma ** 2))
#        return a
#    
#    def VY_verif(x):
#        a=0
#        for k in range(0,N):
#            a=a+MOM[k][1]* np.exp(-((x[0]-CP[k][0]) ** 2 + (x[1]-CP[k][1])  ** 2) / (2 * sigma ** 2))
#        return a
#
#    disp_func_verif = [VX_verif,VY_verif]
#    disp_field_verif= disp_field_space.element(disp_func_verif)
#    deformed_template_verif=deform_op(disp_field_verif)
#    Energy=obj_fun(deformed_template_verif)
#    Energy = Lambda*Energy
#    Energy_data=Energy
#    for k in range(0,N):
#        for j in range(0,N):
#            prod= MOM[k][0]* MOM[j][0] +MOM[k][1]* MOM[j][1]
#            Energy=Energy+0.5*prod*np.exp(-((CP[j][0]-CP[k][0]) ** 2 + (CP[j][1]-CP[k][1])  ** 2) / (2 * sigma ** 2))
#        
#    print("Verification  Iteration= {} ,  Energy= {} ; Data attachment term = {}".format(NumberIter, Energy,Energy_data))
#

        
        
    #--- Computation of the gradient ---
    
    GradCP=copy.deepcopy(CP)
    GradMOM=copy.deepcopy(MOM)
    
    for k in range(0,N):
        # Derivative of disp_field_est with respect to CP[k][0]
        #print("     Iteration= {} ; k= {}".format(NumberIter,k))
        disp_func_grad = [
                          lambda x: ((x[0]-CP[k][0])/(sigma ** 2))*MOM[k][0]* np.exp(-((x[0]-CP[k][0]) ** 2 + (x[1]-CP[k][1])  ** 2) / (2 * sigma ** 2)),
                          lambda x: ((x[0]-CP[k][0])/(sigma ** 2))*MOM[k][1]* np.exp(-((x[0]-CP[k][0]) ** 2 + (x[1]-CP[k][1])  ** 2) / (2 * sigma ** 2))]
                          
        disp_field_grad = disp_field_space.element(disp_func_grad)
        # Derivative of the application of the vector field to template with respect to CP[k][0]
        deform_op_deriv = deform_op.derivative(disp_field_est)(disp_field_grad)
        # Derivation of the energy with respect to CP[0]
        GradCP[k][0]=Lambda*obj_fun.derivative(deformed_template_est)(deform_op_deriv)

        
        # Derivative of disp_field_est with respect to CP[k][1]
        disp_func_grad = [
                          lambda x: ((x[1]-CP[k][1])/(sigma ** 2))*MOM[k][0]* np.exp(-((x[0]-CP[k][0]) ** 2 + (x[1]-CP[k][1])  ** 2) / (2 * sigma ** 2)),
                          lambda x: ((x[1]-CP[k][1])/(sigma ** 2))*MOM[k][1]* np.exp(-((x[0]-CP[k][0]) ** 2 + (x[1]-CP[k][1])  ** 2) / (2 * sigma ** 2))]
        disp_field_grad = disp_field_space.element(disp_func_grad)
        # Derivative of the application of the vector field to template with respect to CP[k][0]
        deform_op_deriv = deform_op.derivative(disp_field_est)(disp_field_grad)
        # Derivation of the energy with respect to CP[0]
        GradCP[k][1]=Lambda*obj_fun.derivative(deformed_template_est)(deform_op_deriv)

        
        # Derivative of disp_field_est with respect to MOM[k][0]
        disp_func_grad = [
                              lambda x: np.exp(-((x[0]-CP[k][0]) ** 2 + (x[1]-CP[k][1])  ** 2) / (2 * sigma ** 2)),
                              lambda x: 0]
        disp_field_grad = disp_field_space.element(disp_func_grad)
        # Derivative of the application of the vector field to template with respect to CP[k][0]
        deform_op_deriv = deform_op.derivative(disp_field_est)(disp_field_grad)
        # Derivation of the energy with respect to CP[0]
        GradMOM[k][0]=Lambda*obj_fun.derivative(deformed_template_est)(deform_op_deriv)
        
        
        # Derivative of disp_field_est with respect to MOM[k][1]
        disp_func_grad = [
        lambda x: 0,
        lambda x: np.exp(-((x[0]-CP[k][0]) ** 2 + (x[1]-CP[k][1])  ** 2) / (2 * sigma ** 2))]
        disp_field_grad = disp_field_space.element(disp_func_grad)
        # Derivative of the application of the vector field to template with respect to CP[k][0]
        deform_op_deriv = deform_op.derivative(disp_field_est)(disp_field_grad)
        # Derivation of the energy with respect to CP[0]
        GradMOM[k][1]=Lambda*obj_fun.derivative(deformed_template_est)(deform_op_deriv)

       # print("             Iteration= {}; k= {}".format(NumberIter,k))
        
        for j in range(0,N):
            if(k!=j):
                GradMOM[k][0]=GradMOM[k][0]+MOM[j][0]*np.exp(-((CP[j][0]-CP[k][0]) ** 2 + (CP[j][1]-CP[k][1])  ** 2) / (2 * sigma ** 2))
                GradMOM[k][1]=GradMOM[k][1]+MOM[j][1]*np.exp(-((CP[j][0]-CP[k][0]) ** 2 + (CP[j][1]-CP[k][1])  ** 2) / (2 * sigma ** 2))
                prod= MOM[k][0]* MOM[j][0] +MOM[k][1]* MOM[j][1]
                GradCP[k][0]=GradCP[k][0]+((CP[k][0]-CP[j][0])/(sigma ** 2))*prod* np.exp(-((CP[j][0]-CP[k][0]) ** 2 + (CP[j][1]-CP[k][1])  ** 2) / (2 * sigma ** 2))
                GradCP[k][1]=GradCP[k][1]+((CP[k][1]-CP[j][1])/(sigma ** 2))*prod* np.exp(-((CP[j][0]-CP[k][0]) ** 2 + (CP[j][1]-CP[k][1])  ** 2) / (2 * sigma ** 2))
            else:
                GradMOM[k][0]=GradMOM[k][0]+MOM[j][0]*np.exp(-((CP[j][0]-CP[k][0]) ** 2 + (CP[j][1]-CP[k][1])  ** 2) / (2 * sigma ** 2))
                GradMOM[k][1]=GradMOM[k][1]+MOM[j][1]*np.exp(-((CP[j][0]-CP[k][0]) ** 2 + (CP[j][1]-CP[k][1])  ** 2) / (2 * sigma ** 2))
        

    
    #--- Linear search ---
    NumberLinSearch=0
    Energy_test=2*Energy
    while (Energy_test>1.0*Energy and NumberLinSearch<NumberMaxLinSearch):
        
        # A : test with stepCP and stepMOM
        CP_testA=copy.deepcopy(CP)
        MOM_testA=copy.deepcopy(MOM)
        for k in range(0,N):
            CP_testA[k][0]=CP[k][0]-stepCP*GradCP[k][0]
            CP_testA[k][1]=CP[k][1]-stepCP*GradCP[k][1]
            MOM_testA[k][0]=MOM[k][0]-stepMOM*GradMOM[k][0]
            MOM_testA[k][1]=MOM[k][1]-stepMOM*GradMOM[k][1]
            
            
        def VX_testA(x):
            a=0
            for k in range(0,N):
                a=a+MOM_testA[k][0]* np.exp(-((x[0]-CP_testA[k][0]) ** 2 + (x[1]-CP_testA[k][1])  ** 2) / (2 * sigma ** 2))
            return a
    
        def VY_testA(x):
            a=0
            for k in range(0,N):
                a=a+MOM_testA[k][1]* np.exp(-((x[0]-CP_testA[k][0]) ** 2 + (x[1]-CP_testA[k][1])  ** 2) / (2 * sigma ** 2))
            return a

        disp_func_testA = [VX_testA,VY_testA]
        disp_field_testA = disp_field_space.element(disp_func_testA)
        deformed_template_testA=deform_op(disp_field_testA)
        Energy_testA=obj_fun(deformed_template_testA)
        Energy_testA = Lambda*Energy_testA
        Energy_data_testA=Energy_testA
        for k in range(0,N):
            for j in range(0,N):
                prod= MOM_testA[k][0]* MOM_testA[j][0] +MOM_testA[k][1]* MOM_testA[j][1]
                Energy_testA=Energy_testA+0.5*prod*np.exp(-((CP_testA[j][0]-CP_testA[k][0]) ** 2 + (CP_testA[j][1]-CP_testA[k][1])  ** 2) / (2 * sigma ** 2))
        
       # print("                   Iteration= {} , Line Search = {} ; Energy= {} ; Data attachment term = {}".format(NumberIter,NumberLinSearch, Energy_testA,Energy_data_testA))


        
        #test with 0.5*stepCP and stepMOM
        CP_testB=copy.deepcopy(CP)
        MOM_testB=copy.deepcopy(MOM)
        for k in range(0,N):
            CP_testB[k][0]=CP[k][0]-0.8*stepCP*GradCP[k][0]
            CP_testB[k][1]=CP[k][1]-0.8*stepCP*GradCP[k][1]
            MOM_testB[k][0]=MOM[k][0]-stepMOM*GradMOM[k][0]
            MOM_testB[k][1]=MOM[k][1]-stepMOM*GradMOM[k][1]
            
            
        def VX_testB(x):
            a=0
            for k in range(0,N):
                a=a+MOM_testB[k][0]* np.exp(-((x[0]-CP_testB[k][0]) ** 2 + (x[1]-CP_testB[k][1])  ** 2) / (2 * sigma ** 2))
            return a
    
        def VY_testB(x):
            a=0
            for k in range(0,N):
                a=a+MOM_testB[k][1]* np.exp(-((x[0]-CP_testB[k][0]) ** 2 + (x[1]-CP_testB[k][1])  ** 2) / (2 * sigma ** 2))
            return a

        disp_func_testB = [VX_testB,VY_testB]
        disp_field_testB = disp_field_space.element(disp_func_testB)
        deformed_template_testB=deform_op(disp_field_testB)
        Energy_testB=obj_fun(deformed_template_testB)
        Energy_testB = Lambda*Energy_testB
        Energy_data_testB=Energy_testB
        for k in range(0,N):
            for j in range(0,N):
                prod= MOM_testB[k][0]* MOM_testB[j][0] +MOM_testB[k][1]* MOM_testB[j][1]
                Energy_testB=Energy_testB+0.5*prod*np.exp(-((CP_testB[j][0]-CP_testB[k][0]) ** 2 + (CP_testB[j][1]-CP_testB[k][1])  ** 2) / (2 * sigma ** 2))
        
       # print("                   Iteration= {} , Line Search = {} ; Energy= {} ; Data attachment term = {}".format(NumberIter,NumberLinSearch, Energy_testB,Energy_data_testB))
        
        #test with stepCP and 0.5*stepMOM
        CP_testC=copy.deepcopy(CP)
        MOM_testC=copy.deepcopy(MOM)
        for k in range(0,N):
            CP_testC[k][0]=CP[k][0]-stepCP*GradCP[k][0]
            CP_testC[k][1]=CP[k][1]-stepCP*GradCP[k][1]
            MOM_testC[k][0]=MOM[k][0]-0.8*stepMOM*GradMOM[k][0]
            MOM_testC[k][1]=MOM[k][1]-0.8*stepMOM*GradMOM[k][1]
            
            
        def VX_testC(x):
            a=0
            for k in range(0,N):
                a=a+MOM_testC[k][0]* np.exp(-((x[0]-CP_testC[k][0]) ** 2 + (x[1]-CP_testC[k][1])  ** 2) / (2 * sigma ** 2))
            return a
    
        def VY_testC(x):
            a=0
            for k in range(0,N):
                a=a+MOM_testC[k][1]* np.exp(-((x[0]-CP_testC[k][0]) ** 2 + (x[1]-CP_testC[k][1])  ** 2) / (2 * sigma ** 2))
            return a

        disp_func_testC = [VX_testC,VY_testC]
        disp_field_testC = disp_field_space.element(disp_func_testC)
        deformed_template_testC=deform_op(disp_field_testC)
        Energy_testC=obj_fun(deformed_template_testC)
        Energy_testC = Lambda*Energy_testC
        Energy_data_testC=Energy_testC
        for k in range(0,N):
            for j in range(0,N):
                prod= MOM_testC[k][0]* MOM_testC[j][0] +MOM_testC[k][1]* MOM_testC[j][1]
                Energy_testC=Energy_testC+0.5*prod*np.exp(-((CP_testC[j][0]-CP_testC[k][0]) ** 2 + (CP_testC[j][1]-CP_testC[k][1])  ** 2) / (2 * sigma ** 2))
        
       # print("                   Iteration= {} , Line Search = {} ; Energy= {} ; Data attachment term = {}".format(NumberIter,NumberLinSearch, Energy_testC,Energy_data_testC))
         
        if (Energy_testA <= Energy_testB and Energy_testA <= Energy_testC):
            Energy_test=Energy_testA
            Energy_data_test=Energy_data_testA
            deformed_template_test=deformed_template_testA
            disp_field_test=disp_field_testA
            CP_test=copy.deepcopy(CP_testA)
            MOM_test=copy.deepcopy(MOM_testA)
        elif (Energy_testB <= Energy_testA and Energy_testB <= Energy_testC):
            Energy_test=Energy_testB
            Energy_data_test=Energy_data_testB
            deformed_template_test=deformed_template_testB
            disp_field_test=disp_field_testB
            CP_test=copy.deepcopy(CP_testB)
            MOM_test=copy.deepcopy(MOM_testB)
            stepCP=0.8*stepCP
        else:
            Energy_test=Energy_testC
            Energy_data_test=Energy_data_testC
            deformed_template_test=deformed_template_testC
            disp_field_test=disp_field_testC
            CP_test=copy.deepcopy(CP_testC)
            MOM_test=copy.deepcopy(MOM_testC)
            stepMOM=0.8*stepMOM
        print("                   Iteration= {} , Line Search = {} ; Energy= {} ; Data attachment term = {}".format(NumberIter,NumberLinSearch, Energy_test,Energy_data_test))
            
        NumberLinSearch=NumberLinSearch+1
        stepCP=0.8*stepCP
        stepMOM=0.8*stepMOM
       
          
    if(NumberLinSearch==NumberMaxLinSearch):
        print("Number maximal of iterations in linear search")
        NumberIter=NumberIterMax
    else:
        Error=((Energy-Energy_test)**2)/(Energy0**2)
        if(Error<Threshold):
            NumberDepThre=NumberDepThre+1
        Energy=Energy_test
        Energy_data=Energy_data_test
        disp_field_est=disp_field_test
        deformed_template_est=deformed_template_test
        CP=copy.deepcopy(CP_test)
        MOM=copy.deepcopy(MOM_test)
        NumberIter=NumberIter+1
        print("Iteration= {} ; Energy= {} ; Data attachment term = {}".format(NumberIter, Energy,Energy_data))
        stepCP=1.*stepCP
        stepMOM=1.2*stepMOM
        

print("Gradient descent done")
template.show('Template');
deformed_template_est.show('Estimated deformed template');
deformed_template.show('Deformed template (Ground truth)');
residual=(template-deformed_template) #/(data+1)
residual.show('Initial differences');
residual=(deformed_template_est-deformed_template) #/(data+1)
residual.show('differences');

# Create projection data by calling the ray transform on the deformed template
data_template = ray_trafo(template)
data_est = ray_trafo(deformed_template_est)
data_template.show(title='Projection template (sinogram)');
data_est.show(title='Projection estimated deformed data (sinogram)');
data.show(title='Projection deformed data (ground truth, sinogram)');
residual_data=(data_template-data) #/(data+1)
residual_data.show('Initial differences differences');
residual_data=(data_est-data) #/(data+1)
residual_data.show('differences');












