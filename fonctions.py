import numpy as np
from scipy import ndimage
import scipy
from pylab import diag, eig

def store_evolution_in(lst):
    """Returns a callback function to store the evolution of the level sets in
    the given list.
    """
    def _store(x):
        lst.append(np.copy(x))

    return _store

def image_contour(image) :
# =============================================================================
#     Gradient par convolution 
# =============================================================================
    contour_kernel_grad = np.zeros_like(image)

    kernel_z = np.zeros((3,3,3))
    kernel_z[0,1,1] = -1
    kernel_z[2,1,1] = 1
    
    kernel_x = np.zeros((3,3,3))
    kernel_x[1,1,0] = -1
    kernel_x[1,1,2] = 1
    
    kernel_y = np.zeros((3,3,3))
    kernel_y[1,0,1] = -1
    kernel_y[1,2,1] = 1    
   
    contour_kernel_grad = np.sqrt((ndimage.filters.convolve(image, kernel_x, mode = 'constant'))**2+((ndimage.filters.convolve(image, kernel_y,mode = 'constant'))**2)+(ndimage.filters.convolve(image, kernel_z,mode = 'constant'))**2)
    return np.array(contour_kernel_grad).astype(np.float32)
#%%   
# =============================================================================
# Fit ellipsoide
# =============================================================================
def mldivide(a, b):
    dimensions = a.shape
    if dimensions[0] == dimensions[1]:
        return scipy.linalg.solve(a, b)
    else:
        return scipy.linalg.lstsq(a, b)[0] 
        
def fit_ellipsoide(XYZ,I):
    # =============================================================================
#    retrouver l'ellipsoïde
#    On défini l'ellipsoïde par son équation : (A²/a² + B²/b² +C²/c²) = 1 
#    On utilise la méthode des moindres carrés : Somme((A²/a² + B²/b² +C²/c² -1 ) min
    # =============================================================================
    N = XYZ.shape[0]
    print(N)
    var=np.zeros((N,9));
    square=np.zeros((N,1)) ;
    XYZ0=np.zeros((N,3));
    dist=np.zeros((N,1)) ;
    cut=np.zeros((N,1)) ;
    C_0 = np.zeros((3,1))
    A = I.copy()
    # définit le nombre d'itérations
    for i in range(0,10):
    # =============================================================================
    # On lance le programme    
    # =============================================================================
        var[:,0] = np.sqrt(A[:,0])*(XYZ[:,0]**2 +XYZ[:,1]**2 -2*XYZ[:,2]**2) 
        var[:,1] = np.sqrt(A[:,0])*(XYZ[:,0]**2 -2*XYZ[:,1]**2 +XYZ [:,2]**2)
        var[:,2] = np.sqrt(A[:,0])*(XYZ[:,0]*XYZ[:,1])
        var[:,3] = np.sqrt(A[:,0])*(XYZ[:,0]*XYZ[:,2])
        var[:,4] = np.sqrt(A[:,0])*(XYZ[:,1]*XYZ[:,2])
        var[:,5] = np.sqrt(A[:,0])*(XYZ[:,0])
        var[:,6] = np.sqrt(A[:,0])*(XYZ[:,1])
        var[:,7] = np.sqrt(A[:,0])*(XYZ[:,2]) 
        var[:,8] = np.sqrt(A[:,0])*(1)
        square[:,0] = np.sqrt(A[:,0])*(XYZ[:,0]**2 +XYZ[:,1]**2 +XYZ[:,2]**2) 
        #% inversion de la relation matricielle
        #% v = ( var' * var ) \ ( var' * square ) ;        
        v,res,ra,s = scipy.linalg.lstsq(var,square)# least_squares_covariance(var,square,I) ;# % poids I_j sur chaque point 
        #% calcul du vecteur u(A/k, B/k, C/k, D/k, E/k, F/k, G/k, H/k, K/k, L/k) (10 composantes) 
        #% à partir de v(U, V, N, M, P, Q, R, S, T) (9 composantes), 
        #% avec un facteur multiplicatif à déterminer k = -(A+B+C)/3.
        u = np.zeros((10,1))
        u[0] = v[0] + v[1] - 1
        u[1] = v[0] - 2 * v[1] - 1
        u[2] = v[1] - 2 * v[0] - 1
        u[3] = v[2]
        u[4] = v[3]
        u[5] = v[4]
        u[6] = v[5]
        u[7] = v[6]
        u[8] = v[7]
        u[9] = v[8]      
        #% definition de la matrice rotation Sk = S/k (au facteur k près)
        #% et calcul des coordonnées du centre de l'ellipsoide C_0 (X_0, Y_0, Z_0)
        Sk = np.array([[float(u[0]), float(u[3])/2, float(u[4])/2] , [ float(u[3])/2, float(u[1]), float(u[5])/2] , [float(u[4])/2, float(u[5])/2, float(u[2])]]);
        TK= np.array([[ float(u[6])] , [float(u[7])] , [float(u[8])]])      
        C_0 = - mldivide(Sk,TK)/2
        # calcul de k et finalisation : vecteurs propres et valeurs propres de S
        k = 1/(np.dot(np.dot(np.transpose(C_0),Sk),C_0) - float(u[9])) 
        S = Sk*float(k)
#        [val,direction] = np.linalg.eigh(S)
#        print("coucou" ,direction)        
        [val,direction] = np.linalg.eig(S)
#        vecpropres = direction              
        demi_axes = [1/np.sqrt(val[0]) , 1/np.sqrt(val[1]) , 1/np.sqrt(val[2])]        
#        print(demi_axes)
        #% calcul d'une distance caractéristique de l'écart à l'ellipsoide, 
        #% en opérant sur les points expérimentaux une translation pour les ramener 
        #% au centre de l'ellipsoide, puis une rotation pour faire coincider avec
        #% les axes principaux, et enfin une homothétie de facteurs (1/a, 1/b, 1/c) print(b[0]*size_pix_x, b[1]*size_pix_y, b[2]*size_pix_z)
        #% pour pouvoir comparer avec la sphère centrée de rayon 1. Delta est la
        #% racine carrée (divisée par le nb de points) de la somme des carrés des distances 
        #% des points expérimentaux modifiées à la sphère.
        #
        #% translation au centre de l'ellipsoide
        XYZ0[:,0] = XYZ[:,0]-C_0[0]
        XYZ0[:,1] = XYZ[:,1]-C_0[1]
        XYZ0[:,2] = XYZ[:,2]-C_0[2]
        #% rotation vers les axes principaux puis homothétie vers la sphère de rayon 1
        XYZ0=np.dot(XYZ0,direction);        
    #    print('blurp',np.any(XYZ0==0), XYZ0.shape)
        XYZ0[:,0] = XYZ0[:,0]/demi_axes[0]
        XYZ0[:,1] = XYZ0[:,1]/demi_axes[1]
        XYZ0[:,2] = XYZ0[:,2]/demi_axes[2]
    #    print('blurp',np.any(XYZ0==0), XYZ0.shape)
        #%calcul des distances de chaque point à la sphère et calcul de Delta
        for j in range(0,N) :    
            dist[j] = abs(1-np.sqrt(XYZ0[j,0]**2+XYZ0[j,1]**2+XYZ0[j,2]**2)) 
            cut[j] = ((0.1-dist[j])/abs(0.1-dist[j])+1)/2 
            A[j] = I[j]*cut[j] 
#    Delta = np.linalg.norm(dist)/N
#    Nb = np.dot(cut.T,cut)    
#    Deltab = np.linalg.norm(cut*dist)/float(Nb)
    return demi_axes,direction,C_0

#%%
def matrix_inertia (stack,barycentre_z,barycentre_x,barycentre_y,size_pix) : 
    """définition de la matrice d'inertie de la bille. On diagonalise et défini
    sa matrice de passage."""
    size_pix_x,size_pix_y,size_pix_z = size_pix   
    xy=0;yz=0;xz=0;x2=0;y2=0;z2=0;N=0;
#    M=0;
    volume_voxel = (size_pix_x*size_pix_y*size_pix_z) 
    masse_bille = 0 ; densite = 1
    for k in range(stack.shape[0]):
        for i in range(stack.shape[1]):
            for j in range(stack.shape[2]) :
#                M=M+1
                if stack[k,i,j]>0 :
                    z = (k-barycentre_z)*size_pix_z
                    x = (i-barycentre_x)*size_pix_x
                    y = (j-barycentre_y)*size_pix_y
                    x2=x2+x**2
                    y2=y2+y**2
                    z2=z2+z**2
                    xy=xy+x*y
                    xz=xz+x*z
                    yz=yz+y*z
                    N=N+1 # N est le nombre de volume élémentaire composant la bille
     
    x2_y2=(x2+y2)*volume_voxel        
    y2_z2=(y2+z2)*volume_voxel
    x2_z2=(x2+z2)*volume_voxel
    
    xy=-xy*volume_voxel
    xz=-xz*volume_voxel
    yz=-yz*volume_voxel
    masse_bille = densite*N*volume_voxel #Densité * nombre de dV * dV
    matri = np.array([(x2_y2,xz,yz),(xz,y2_z2,xy),(yz,xy,x2_z2)], dtype = float)   
    P=eig(matri)[1] # Matrice de passage de la base définie initiallement à celle ou la matrice est diagonale 
    D=diag(eig(matri)[0]) # matrice diagonale de matri
    #print(matri-P*D*inv(P)) #vérification qu'on a bien les bonnes matrices
    return (D, P, masse_bille) #a,b et c sont les éléments diagonaux de la matrices d'intertie et les autres termes

def ellipsoide_parametres(D,masse_bille):
    a = 0; b = 0; c = 0; d1 =0; d2 = 0; d3 = 0;
    d1 = (5/(2*masse_bille))*D[0][0]
    d2 = (5/(2*masse_bille))*D[1][1]
    d3 = (5/(2*masse_bille))*D[2][2]

    a = np.sqrt(-d1+d2+d3)
    b = np.sqrt(d1-d2+d3)
    c = np.sqrt(d1+d2-d3)
    volume =  (4/3)*np.pi*a*b*c  
    return (a,b,c,volume)

def generate_ellipsoide(a,b,c,Z0,X0,Y0,theta,psi,phi,shape) : 
    #a , b, c sont les demis axes de l'ellispoide
    # x0, y0, z0 centre de l'ellipsoide    
    #% angles d'Euler (en degrés) définissant les axes de l'ellipsoide
    c1 = np.cos(psi*np.pi/180);
    s1 = np.sin(psi*np.pi/180);
    c2 = np.cos(theta*np.pi/180);
    s2 = np.sin(theta*np.pi/180);
    c3 = np.cos(phi*np.pi/180);
    s3 = np.sin(phi*np.pi/180);  
  
    R= np.zeros((3,3))
    R[0,:] = [s1*s2  , c1*c3-s1*s3*c2 , -c1*s3-s1*c3*c2 ] 
    R[1,:] =  [-c1*s2  , s1*c3+c1*c2*s3 , -s1*s3+c1*c2*c3 ]
    R[2,:] =  [ c2 , s2*s3 , s2*c3 ] 
    
    xyz = np.zeros((79600,3))  
    for i in range(1,200) : 
        for j in range(1,401) : 
            k=(i-1)*400+j;       
            k=k-1
            if k <0 : continue;
            xyz[k,0]=a*np.sin(np.pi*i/200)*np.cos(np.pi*j/200);
            xyz[k,1]=b*np.sin(np.pi*i/200)*np.sin(np.pi*j/200);
            xyz[k,2]=c*np.cos(np.pi*i/200);
    #    print(k)
    RXYZ = np.zeros((79600,3))
    for k in range(0,79600) : 
        for n in range(0,3): 
            RXYZ[k,n] = R[n,0]*xyz[k,0] + R[n,1]*xyz[k,1] + R[n,2]*xyz[k,2]
    
    #    XYZ = np.zeros((79600,3))
    ellipsoide_generee_3D = np.zeros(shape)
    for k in range(0,79600) : 

        if int(RXYZ[k,0]+Z0) < 0 : z = 0 ;
        elif int(RXYZ[k,0]+Z0) >= shape[0] : z = shape[0]-1; #x = shape[1]    
        else : z = int(RXYZ[k,0]+Z0)
        
        if int(RXYZ[k,1]+X0) < 0 : x = 0;
        elif int(RXYZ[k,1]+X0) >= shape[1] : x = shape[1]-1# y = shape[2]
        else : x = int(RXYZ[k,1]+X0)
        
        if int(RXYZ[k,2]+Y0) < 0 : y = 0;
        elif int(RXYZ[k,2]+Y0) >= shape[2] : y = shape[2]-1;#z = shape[0]
        else : y = int(RXYZ[k,2]+Y0)           
        ellipsoide_generee_3D[z,x,y] = 1      
    return ellipsoide_generee_3D


def angle(P):
#    on choisi arbitrairement teta positif
    teta = np.arccos(P[2][0])
    if teta < 0 :
        teta = -teta
#        on a donc :
    sin_psi = P[0,0]/np.sin(teta)
    cos_psi = -P[1,0]/np.sin(teta)      
    psi = np.sign(sin_psi)*np.arccos(cos_psi)
   
    sin_phi = P[2,1]/np.sin(teta)
    cos_phi = P[2,2]/np.sin(teta)
    phi = np.sign(sin_phi)*np.arccos(cos_phi)
    
    a = np.cos(psi)*np.cos(phi)-np.sin(psi)*np.cos(teta)*np.sin(phi)
    b = np.sin(psi)*np.cos(phi)+np.cos(psi)*np.cos(teta)*np.sin(phi)
    c = np.sin(teta)*np.sin(phi) 
    if round(a,3) != round(P[0,1],3) and round(b,3) != round(P[1,1],3) and round(c,3) != round(P[2,1],3) : 
        print("marche pas...")
 
    teta = teta *180/np.pi
    psi = psi *180/np.pi
    phi = phi*180/np.pi
    return(teta,psi,phi,P)

def matrice_passage (a_cont,b_cont,c_cont,P_cont):
    P= np.zeros_like(P_cont);
    a,b,c = 0,0,0
    if max(list(abs(P_cont[0,:]))) == abs(P_cont[0,0]): 
        P[:,0] = P_cont[:,0]; a = a_cont
    if max(list(abs(P_cont[0,:]))) == abs(P_cont[0,1]): 
        P[:,0] = P_cont[:,1]; a = b_cont
    if max(list(abs(P_cont[0,:]))) == abs(P_cont[0,2]): 
        P[:,0] = P_cont[:,2]; a = c_cont
       
    if max(list(abs(P_cont[1,:])))== abs(P_cont[1,0]): 
        P[:,1] = P_cont[:,0]; b = a_cont
    if max(list(abs(P_cont[1,:]))) == abs(P_cont[1,1]): 
        P[:,1] = P_cont[:,1]; b = b_cont
    if max(list(abs(P_cont[1,:]))) == abs(P_cont[1,2]): 
        P[:,1] = P_cont[:,2]; b = c_cont
        
    if max(list(abs(P_cont[2,:]))) == abs(P_cont[2,0]): 
        P[:,2] = P_cont[:,0]; c = a_cont
    if max(list(abs(P_cont[2,:]))) == abs(P_cont[2,1]): 
        P[:,2] = P_cont[:,1]; c = b_cont
    if max(list(abs(P_cont[2,:]))) == abs(P_cont[2,2]): 
        P[:,2] = P_cont[:,2]; c = c_cont
        
    if P[0,0]== P[0,1] or P[0,0] == P[0,2] or P[0,1]==P[0,2]:  
        P = P_cont; a = a_cont ; b = b_cont ; c = c_cont
        etat = "impossible à trier"
    else : etat = "tout est ok"
   
    if round(P[1,1]*P[2,2]-P[1,2]*P[2,1],3) != round(P[0,0],3) : 
        P[:,0] = -P[:,0]       
    if P[2,0] < 0 :
        P[:,0] = -P[:,0] ; P[:,1] = -P[:,1]
    if round(P[1,1]*P[2,2]-P[1,2]*P[2,1],3) == round(P[0,0],3) : 
        print("Le repère est définitivement direct!") 
        etat = etat + " et le repère est direct!"
    else : etat = etat + " et on a pas réussi à le mettre direct..."
    return a,b,c,P,etat