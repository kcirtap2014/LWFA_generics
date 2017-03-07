import numpy as np
import pdb
from scipy.constants import e, c, m_e, epsilon_0
import random

def cm2inch(value):
    """
    Convert cm 2 inch for matplotlib drawing

    Parameters:
    -----------
    cm: float
        value in cm

    Returns:
    --------
    inch: float
        value in inch
    """

    return value/2.54

def randintGenerator( number, pool ):
    """
    Returns an array of random int

    Parameters:
    -----------
    number: int
        Size of the returned array

    pool: array
        pool to choose from

    Returns:
    --------
    randomInt: an array of int
        an array of random numbers
    """

    randomInt =random.sample( range(pool[0] , pool[1]), number )

    return randomInt

def wp( ne ):
    """
    calculate plasma frequency
    """
    return np.sqrt(e**2*ne/(epsilon_0*m_e))

def find_nearest( array, value ):
    """
    find the nearest value and return the index

    Parameters:
    -----------
    array: 1D numpy array
        the array that we want to look up

    value: float
        the value we are looking for

    Returns:
    --------
    idx: int
        index corrsponding to the nearest value

    """
    idx = (np.abs(array-value)).argmin()

    return idx


def rungeKutta4( f, x0, y0, h, xfinal, g = None, z0 = None, invert_scale = False,
                var1 = None, var2 = None, var3 = None):
    """
    This function resolves an ODE using Runge Kutta 4 method.
    an ODE has the form of dy/dx = f(x,y).

    Parameters:
    -----------
    f: function
        the RHS function of an ODE, containing variables x and y

    x0: float
        initial value for x

    y0: float
        initial value for y

    h: float
        step size

    xfinal: float
        the final x value

    g: function
        optional: use only if you are solving a 2nd order ODE where you
        decompose the 2nd order ODE into 2 1st order ODE, this function
        represents the 2nd 1st ODE

    z0: float
        initial value for z

    invert_scale: boolean
        Calculate from postive to negative if True. Default: False

    asquare: an array of float
        a variable that is depending of x. Example usage: laser pulse in LWFA

    kp: an array of float
        a parameter that can be dependant of x.
        Example usage: laser pulse in LWFA

    gamma_g: an array of float
        a parameter that can be dependant of x.
        Example usage: laser pulse in LWFA

    Returns:
    --------
    x_array : array of floats
        array of x values

    y_array : array of floats
        array of y values

    z_array : array of floats
        array of z values [optional]
    """

    x = x0
    y = y0

    # to store the x,y values
    x_array = []
    y_array = []

    if g is not None:
        z = z0
        z_array = []

    index = 0

    # Define conditions, depends if invert_scale is True
    if invert_scale:
        condition = x>xfinal
    else:
        condition = x<xfinal

    while condition:
        # save x and y values
        x_array.append( x )
        y_array.append( y )

        #Runge Kutta method
        if g is not None:
            # save z values
            z_array.append( z )
            try:
                var1vec = var1[index]
            except TypeError:
                var1vec = 0

            k1 = h*f( x, y, z )
            l1 = h*g( x, y, z, var1vec, var2, var3 )
            k2 = h*f( x + h/2, y + k1/2, z + l1/2 )
            l2 = h*g( x + h/2, y + k1/2, z + l1/2, var1vec, var2, var3)
            k3 = h*f( x + h/2, y + k2/2, z + l2/2 )
            l3 = h*g( x + h/2, y + k2/2, z + l2/2, var1vec, var2, var3 )
            k4 = h*f( x + h, y + k3, z + l3)
            l4 = h*g( x + h, y + k3, z + l3, var1vec, var2, var3)
            z = z + (1./6)* ( l1 + 2*l2 + 2*l3 + l4 )

        else:
            k1 = h*f( x, y )
            k2 = h*f( x + h/2, y + k1/2 )
            k3 = h*f( x + h/2, y + k2/2 )
            k4 = h*f( x + h, y + k3)

        y = y + (1./6)* ( k1 + 2*k2 + 2*k3 + k4 )

        # incrementation
        x += h
        index += 1

        # redefine condition
        if invert_scale:
            condition = x>xfinal
        else:
            condition = x<xfinal

    ans = [ x_array, y_array ]

    if g is not None:
        ans += [z_array]

    return ans

def xi(z, vg, t):
    """
    Define the comiving coordinates

    Parameters:
    -----------
    z : an array of float
        an array of position

    vg: float
        fluid velocity

    t: an array of float
        an array of time

    Returns:
    --------
    Array of float
        comoving coordinates

    """
    return np.array(z - vg*t)

def ioniz_probability( Edc, Uion ):
    """
    Calculate the probability of ionization
    based on the ADK model. The ionization rate is given for a Hydrogen-like
    system by the Landau-Lifshitz DC tunneling ionization formula

    Parameters:
    -----------
    Edc : 1D array or float value
        external electric field

    Uion: 1D array or float value
        ionization potential of any atom. A control on the size is done at the
        beginning

    Returns:
    --------
    rate: 1D array
        rate of ionisation
    """

    alpha = 1./137
    re = (e**2)/(4*np.pi*epsilon_0*m_e*c**2)
    omega_a = alpha**3*c/re #atomic unit frequency
    Ea = m_e*c**2*alpha**4/(re*e)
    UH = 13.6 #in eV

    if Uion.size > 1:
        rate = [[] for i in xrange(len(Uion))]

        for index_i,ion in enumerate(Uion):
            for index in xrange(len(Edc)):
                if Edc[index]>Ea:
                    rate[index_i].append(4*omega_a*(Ea/Edc[index])*(ion/UH)**(2.5)*np.exp(-(2./3)\
                    *(Ea/Edc[index])*(ion/UH)**1.5))
                else:
                    rate[index_i].append(0.0)

    else:
        if Edc>Ea:
            rate = 4*omega_a*(Ea/Edc)*(Uion/UH)**(2.5)*np.exp(-(2./3)\
            *(Ea/Edc)*(Uion/UH)**1.5)
        else:
            rate = 0.0

    return rate

def vp(z,t, kp):
    """
    return the phase velocity in a density downramp

    Parameters:
    -----------
    z: 1D array
        longitudinal position

    t: 1D array
        time

    kp: 1D array
        kp(z)

    Returns:
    --------
    vp: 1D array
        vp(z,t)
    """
    vg = c
    vp = []
    grad_kp = []

    for i in range(0,len(kp)-1):
        grad_kp.append((kp[i+1] - kp[i])/(z[i+1] - z[i]))
    grad_kp.append(0.0)

    for i in xrange(len(kp)):
        vp.append(vg/(1. + (z[i] - vg*t[i])*grad_kp[i]/kp[i]))

    return vp

def nonlinearLambdaP(lambda_p ,EmaxsE0):
    """
    Returns the nonlinear plasma wavelength. Eq. 25 of Esarey Review's paper

    Parameters:
    -----------
    lambda_p : float
        linear plasma wavelength

    EmaxsE0 : float
        Emax/E0

    Returns:
    --------
    lambda_np: float
        nonlinear plasma wavelength
    """

    if EmaxsE0<1:
        lambda_np = lambda_p*(1 + 3*EmaxsE0**2/16)
    else:
        lambda_np = lambda_p*(2./np.pi)*(EmaxsE0 + 1./EmaxsE0)

    return lambda_np

def depletionLength(lambda_p, lambda0, a0):
    """
    Returns the laser depletion length.

    Parameters:
    -----------
    lambda_p : float
        linear plasma wavelength

    lambda0 : float
        laser wavelength

    a0 : float
        a0

    Returns:
    --------
    Lpd: float
        laser depletion length
    """
    if a0**2>1:
        Lpd = (lambda_p**3/lambda0**2)*a0
    else:
        Lpd = (lambda_p**3/lambda0**2)/a0**2

    return Lpd


def dephasingLength(lambda_np, gamma_p, EmaxsE0):
    """
    Returns the nonlinear plasma wavelength. Eq. 25 of Esarey Review's paper

    Parameters:
    -----------
    lambda_np : float
        nonlinear plasma wavelength

    EmaxsE0 : float
        Emax/E0

    gamma_p: float
        Lorentz factor of the wake

    Returns:
    --------
    Ld: float
        dephasing length
    """

    if EmaxsE0<1:
        Ld = gamma_p**2*lambda_np*2/np.pi
    else:
        Ld = gamma_p**2*lambda_np/2

    return Ld

def s_dephasingLength(lambda_p, lambda0, a0):
    """
    Returns the nonlinear plasma wavelength. Eq. 25 of Esarey Review's paper

    Parameters:
    -----------
    lambda_p : float
        plasma wavelength

    lambda0 : float
        laser wavelength

    a0: float
        laser vector potential

    Returns:
    --------
    Ld: float
        dephasing length
    """

    if a0**2<1:
        Ld = (lambda_p**3/lambda0**2)
    else:
        Ld = (lambda_p**3/lambda0**2)*a0

    return Ld

def alaser( a0, laser_ctau, z, t, k0, omega0, x, high_frequency = False ):
    """
    Define the laser pulse

    Parameters:
    -----------
    a0: float
        max normalized vector potential

    laser_tau: float
        laser c_tau in 1/e^2

    z: array of float
        real position in z

    t: array of float
        real position in time. for temporal profile

    k0: float
        laser wavenumber

    omega0: float
        laser frequency

    x: array of float
        comoving coordinates, xi

    Returns:
    --------
    the expression of the laser pulse with a Gaussian temporal profile

    """
    # phase shift with the reference of laser central position
    asquare = a0**2*np.exp(-x**2/laser_ctau**2)
    a = a0**2*np.exp(-x**2/laser_ctau**2)

    if high_frequency:
        asquare*=np.cos(k0*z - omega0*t)**2
        a *= np.cos(k0*z - omega0*t)
    return a, asquare

def findiff(y,h):
    """
    Finite difference method

    Parameters:
    -----------
    y : an array of float
        y variable

    h : float
        finite difference step size

    Returns:
    --------
    deriv_y : an array of float
        the derived y variable
    """

    return (np.array(y[:-1]) - np.array(y[1:]))/h

def findRoot( y, x ):
    """
    finds the zero crossing or the roots of the input signal, the input signal
    should be somewhat smooth.

    Parameters:
    -----------
    x: 1D numpy array
        values in x-coordinates

    y: 1D numpy array
        values in y-coordinates

    Returns
    -------
    roots : 2D numpy array
    """
    #displaying the sign of the value
    l = np.shape(y)[0]
    s = np.sign(y)
    index = []

    for i in xrange ( l-1 ):
        if (s[i+1]+s[i] == 0 ):
            # when the sum of the signs ==0, that means we hit a 0
            index.append(i)

    roots = np.take( x , index ).tolist()
    lrz = len(roots)
    # if there's only one root found,
    # there should be an end to it, we consider the min z
    # as the the limit

    # insert a z value at the first index
    if lrz == 1:
        roots.insert( 0, min(x) )
    # if length of root is not pair, we remove the first value
    if np.shape(roots)[0]%2 != 0:
        roots = np.delete( roots, 0 )

    return roots

def hamiltonianSep ( phi, a, gamma_g ):
    """
    define the hamiltonian of the separatrix

    Parameters:
    -----------
    phi : 1D array
        the wakefield potential

    a : float
        normalized laser vector potential

    gamma_g: float
        gamma_g of the laser pulse

    Returns:
    --------
    H: float
        Hamiltonian of the separatrix
    """
    phi_min = np.min(phi)

    # we take the index_min close to the laser pulse, which means the biggest
    # index_min
    index_min = np.argmin(phi)

    return np.sqrt(1 + a[np.max(index_min)])/gamma_g - phi_min

def energySep( x, a, gamma_g, kp):
    """
    give the minimum energy necessary for trapping.

    Parameters:
    -----------
    phi : 1D array
        the wakefield potential

    a : float
        normalized laser vector potential

    gamma_g: 1D array
        gamma_g of the laser pulse

    Returns:
    --------
    energy: 1D array
        energy threshold for trapping
    """


    def f(x, y, z):
        return z

    def g(x, y, z, sqrta, kp, gamma_g):
        return (kp*gamma_g)**2*(beta_g/np.sqrt(1-(1 + sqrta**2)/\
                (gamma_g*( 1 + y ))**2)-1)

    E = [[] for i in xrange(len(gamma_g))]
    phi_min =[[] for i in xrange(len(gamma_g))]
    gamma_g_list = []

    for index_a, a_value in enumerate(a):
        for index_g, g_value in enumerate(gamma_g):

            beta_g = np.sqrt(1.-1./g_value**2)
            ans = rungeKutta4( f, max(x[index_g]), 0.0,
                                        x[index_g][1] - x[index_g][0],
                                        min(x[index_g]), g = g,
                                        z0 = 0.0, invert_scale = True,
                                        var1 =np.sqrt(a_value[index_g]),
                                        var2 = kp[index_g],
                                        var3 = g_value)

            phi = ans[1]
            Hsep = hamiltonianSep ( phi, a_value[index_g], g_value )
            phi_min[index_g].append(np.min(phi))

            uz = uzSep( Hsep, beta_g, g_value )
            E[index_g].append((m_e*c**2)*(np.sqrt(1 + uz**2)-1))
        gamma_g_list.append(gamma_g)
        pdb.set_trace()

    return E, phi_min, gamma_g_list

def hamiltonian( phi, uz, a, beta_g):
    """
    Define the Hamiltonian

    Parameters:
    -----------
    x: float
        xi position

    phi: float
        phi(x)

    uz: float
        uz(x)

    ux: float
        ux(x)

    a: float
        a(x)

    beta_g: float
        beta_g(x)

    Returns:
    --------
    Hamiltonian at instant x:
        sqrt(1+(ux + a)^2 +uz^2) - phi - beta_g*uz

    """
    return np.sqrt(1 + (a) + uz**2) - phi - beta_g*uz

def trapped_hamiltonian( phi, gamma_g ):
    """
    Define the Hamiltonian for trapped particles

    Parameters:
    -----------
    phi: float
        phi at a precise moment

    gamma_g: float
        gamma_g at a precise moment

    Returns:
    --------
    Hamiltonian which is less than Hamiltonian at separatrix:
        1./gamma_g - phi

    """
    return 1./gamma_g - phi

def ionized_hamiltonian( phi ):
    """
    Define the Hamiltonian for ionized electrons

    Parameters:
    -----------
    phi: float
        phi at a precise moment

    Returns:
    --------
    Hamiltonian which is less than Hamiltonian at separatrix:
        1. - phi

    """
    return 1. - phi

def uzAtOneInstantPlus( beta_g, gamma_g, H, phi, asquare):
    """
    calculate uz -

    Parameters:
    -----------
    beta_g: float
        beta of the fluid

    gamma_g: float
        Lorentz factor of the fluid

    H: float
        Hamiltonian

    phi: float
        phi

    asquare: float
        laser normalized vector potential square

    Returns:
    --------
        uz minus
    """

    Hphi = H + phi

    return beta_g*gamma_g**2*(Hphi) + \
            gamma_g*np.sqrt(gamma_g**2*(Hphi)**2 - (1 + asquare))

def uzAtOneInstantMinus( beta_g, gamma_g, H, phi, asquare):
    """
    Calulate uz +

    Parameters:
    -----------
    beta_g: float
        beta of the fluid

    gamma_g: float
        Lorentz factor of the fluid

    H: float
        Hamiltonian

    phi: float
        phi

    asquare: float
        laser normalized vector potential square

    Returns:
    --------
        uz plus
    """

    Hphi = H + phi

    return  beta_g*gamma_g**2*(Hphi) - \
                    gamma_g*np.sqrt(gamma_g**2*(Hphi)**2 - (1 + asquare))

def uzSep( Hsep, beta_g, gamma_g):
    """
    Calculates the longitudinal momentum of the separatrix

    Parameters:
    -----------
    Hsep: float
        Hamiltonian of the separatrix

    beta_g: float
        beta_g of the laser pulse

    gamma_g: float
        gamma_g of the laser pulse

    Returns:
    --------
    uz_sep: float
        longitudinal momentum of the electron at separatrix
    """

    return (beta_g*gamma_g**2)*Hsep - gamma_g*np.sqrt((gamma_g*Hsep)**2 - 1)

def phase_space( x, phi, a, beta_g, gamma_g, Hsep, num_hamiltonian = 11,
    ionization = False):
    """
    Electron trajectories in phase space. For each position in the phase space,
    we solve for all uz.

    Parameters:
    -----------
    x : array of floats
        xi

    phi: array of floats
        phi

    beta_g: float
        beta_g (supposed constant throughout the propagation)

    gamma_g: float
        gamma_g (supposed constant throughout the propagation)

    Hsep: float
        Hamiltonian of the separatrix

    num_hamiltonian: int
        number of hamiltonians to be analyzed

    ionization: boolean
        if True, we are in the case for ionization. Default: False

    Returns:
    --------
    H_chosen: an array of floats
        num_hamiltonian of chosen hamiltonians

    uz_trapped_plus_array: an array of floats of size (num_hamiltonian, len(x))
        uz + corresponding to trapped hamiltonian

    uz_trapped_minus_array: an array of floats
        uz - corresponding to trapped hamiltonian

    uz_sep_max: an array of floats
        uz + corresponding to hamiltonian of separatrix

    uz_sep_min: an array of floats
        uz - corresponding to hamiltonian of separatrix

    uz_rest_max: an array of floats
        uz + corresponding to hamiltonian of the electrons initially at rest

    uz_rest_min: an array of floats
        uz - corresponding to hamiltonian of the electrons initially at rest

    trapped_index: array of int
        index that corresponds to the trapping phase of electrons
    """
    H_array = []
    trapped_index = []

    for i in xrange(len(x)):
        if ionization:
            H_temp = ionized_hamiltonian( phi[i] )
            if np.logical_and(H_temp < Hsep, a[i] > 1.37) : #have to put a constant here
                trapped_index.append(i)

        else:
            H_temp = trapped_hamiltonian( phi[i], gamma_g )
            if H_temp < Hsep:
                trapped_index.append(i)

        H_array.append(H_temp)

    if ionization:
        trapped_index = np.arange(np.min(trapped_index), np.max(trapped_index))

    H_chosen = np.linspace(min(H_array), Hsep, num_hamiltonian)
    H_chosen = H_chosen[-1:0:-1]

    stride = 1
    uz_trapped_plus_array = [[] for i in xrange(len(H_chosen))]
    uz_trapped_minus_array = [[] for i in xrange(len(H_chosen))]

    uz_sep_max = []
    uz_sep_min = []
    uz_rest_max = []
    uz_rest_min = []
    Hrest =  1.

    for i in xrange(len(x)):
        # Calculation for the particle at rest
        uz_rest_plus = uzAtOneInstantPlus( beta_g,
                        gamma_g, Hrest, phi[i], a[i])
        uz_rest_minus = uzAtOneInstantMinus( beta_g,
                        gamma_g, Hrest, phi[i], a[i] )
        uz_rest_max.append( uz_rest_plus  )
        uz_rest_min.append( uz_rest_minus )

        #Calculation for the particle that forms the separatrix
        uz_sep_plus = uzAtOneInstantPlus( beta_g,
                        gamma_g, Hsep, phi[i], a[i])
        uz_sep_minus = uzAtOneInstantMinus( beta_g,
                        gamma_g, Hsep, phi[i], a[i] )
        uz_sep_max.append( uz_sep_plus )
        uz_sep_min.append( uz_sep_minus )


    for index, part in enumerate(H_chosen[::stride]):
        for i in xrange(len(x)):
            uz_temp_plus = uzAtOneInstantPlus( beta_g,
                            gamma_g, part, phi[i], a[i])
            uz_temp_minus = uzAtOneInstantMinus( beta_g,
                            gamma_g, part, phi[i], a[i] )

            if uz_temp_plus> np.nanmax(uz_sep_min):
                uz_trapped_plus_array[index].append(uz_temp_plus)
            else:
                uz_trapped_plus_array[index].append(np.nan)

            uz_trapped_minus_array[index].append(uz_temp_minus)

    return H_chosen, uz_trapped_plus_array, uz_trapped_minus_array, \
            uz_sep_max, uz_sep_min, uz_rest_max, uz_rest_min, trapped_index

def binary_search(target, x, left, right):
    """
    Recursive binary search algorithm

    Parameters:
    -----------
    target : int
        the value that we are looking for in the array

    x : int
        the array where we are looking at

    Returns :
    ---------
    index : int
        index of the key in the x array
    """

    m = int((left + right)/2)

    if left!=right:
        if target == x[m]:
            index = m
        elif target<x[m]:
            index = binary_search( target, x, left, m )
        elif target>x[m]:
            index = binary_search( target, x, m+1, right )
    else:
        index = -1

    return index

def trajectory_sim(Ezfield, Eyfield, Exfield, xlong, xtrans, gamma_g, dt,
                    zinit, pinit, num_steps, num_particles = 1 ):
    """
    Using a random generator to initialize the initial position of the electron,
    then simulate the trajectory of these particles in presence of the Efields.
    Resolution using RungeKutta

    Parameters:
    -----------
    Ezfield: an array of floats
        Longitudinal Wakefield

    Eyfield: an array of floats
        Laser field

    Exfield: an array of floats
        Transverse Wakefield

    xlong: D numpy array
        longitudinal x values

    xlong: D numpy array
        transverse x values

    gamma_g: float (for now)
        group velocity of the laser

    dt: float
        time step

    zinit: 2D numpy array of shape (1,2)
        initial values for position

    pinit: 2D numpy array of shape (1,2)
        initial values for moment

    num_steps: int
        number of time steps

    num_particles: int
        number of particles

    Returns:
    --------
    t_array: a 2D array of floats of shape (num_steps, num_particles )
        indicates the time

    p_array: a 2D array of floats of shape (num_steps, num_particles )
        indicates the moment

    z_array: a 2D array of floats of shape (num_steps, num_particles )
        indicates the position
    """

    #generate position of an electron
    #for part in xrange(num_particles):
    #    randint()
    def f(t, z):
        indexlong = int((z[0] - np.min(xlong))/long_dynamics*len(xlong))
        indextransx = int((z[1] - np.min(xtrans))/trans_dynamics*len(xtrans))

        xlongdiff = np.absolute( xlong[indexlong+1]- xlong[indexlong] )
        xtransxdiff = np.absolute( xtrans[indextransx+1]- xtrans[indextransx] )

        Ezpartlong = Ezfield[indextransx, indexlong+1]*( \
                     xlong[indexlong +1] - z[0])/(xlongdiff) + \
                     Ezfield[indextransx, indexlong]\
                     *(z[0] - xlong[indexlong] )/(xlongdiff)

        Ezparttrans = Ezfield[ indextransx + 1, indexlong ]*( \
                     xtrans[indextransx+1] - z[1])/(xtransxdiff) + \
                     Ezfield[indextransx, indexlong]*( \
                     z[1]- xtrans[indextransx] )/(xtransxdiff)

        Expartlong = Exfield[indextransx, indexlong+1]*( \
                    xlong[indexlong +1] - z[0])/(xlongdiff) + \
                    Exfield[indextransx, indexlong]\
                    *(z[0] - xlong[indexlong] )/(xlongdiff)

        Exparttrans = Exfield[ indextransx + 1, indexlong ]*( \
                     xtrans[indextransx + 1] - z[1])/(xtransxdiff) + \
                     Exfield[indextransx, indexlong]*( \
                     z[1]- xtrans[indextransx] )/(xtransxdiff)

        E = np.array([(Ezpartlong + Ezparttrans)/2  ,
                        (Expartlong + Exparttrans)/2])
        return -e*E

    def g(t, p, gamma_g):
        # first index corresponds to the longitudinal direction
        # 2nd index corresponds to the transverse direction

        gamma = np.sqrt(1 + (p[0]**2 + p[1]**2)/(m_e*c)**2)
        ptrans = p[1]/(m_e*gamma)
        plong = p[0]/(m_e*gamma) - c*(1 - 0.5/gamma_g**2)

        return np.array([plong, ptrans])

    z_array = [] # for position
    t_array = [] # for time
    p_array = [] # for moment

    #Initialization
    t = 0.0
    p = p_old = pinit
    z = zinit
    h = dt
    index = 0
    long_dynamics = np.max(xlong)-np.min(xlong)
    trans_dynamics = np.max(xtrans)-np.min(xtrans)
    moreThanGammaG = 0
    lessThanGammaG = 0
    k = 1

    while index<num_steps:
        try:
            # save x, y, z values
            t_array.append( t )
            p_array.append( p )
            z_array.append( z )
            # Look up the field that corresponds to the position

            #Runge Kutta method
            k1 = h*f( t, z )
            l1 = h*g( t, p, gamma_g )
            k2 = h*f( t + h/2,  z )
            l2 = h*g( t + h/2, p + k1/2, gamma_g )
            k3 = h*f( t + h/2,  z )
            l3 = h*g( t + h/2, p + k2/2, gamma_g )
            k4 = h*f( t + h, z )
            l4 = h*g( t + h, p + k3, gamma_g )

            p = p + (1./6)* ( k1 + 2*k2 + 2*k3 + k4 )
            z = z + (1./6)* ( l1 + 2*l2 + 2*l3 + l4 )

            #Adaptative time step
            dpdt = (p[0] - p_old[0])/h
            if p[0]/(m_e*c) >= gamma_g and moreThanGammaG == 0:
                h *= 5
                moreThanGammaG += 1


            if dpdt < 0 and lessThanGammaG == 0:
                h /= 5
                lessThanGammaG += 1

            # incrementation
            t += h
            index += 1
            p_old = p

        except IndexError:
            break

    return t_array, z_array, p_array

def m_trajectory_sim(Ezfield, Eyfield, xlong, gamma_g, dt, zinit, pinit, tinit,
                    num_steps, dimension ="1D", xtrans = None, Exfield = None,
                    latency = None, num_particles = 1 , diag_period = 100,
                    accel = 10,  alreadyAccel = None):
    """
    Using a random generator to initialize the initial position of the electron,
    then simulate the trajectory of these particles in presence of the Efields.
    Resolution using RungeKutta

    Parameters:
    -----------
    Ezfield: an array of floats
        Longitudinal Wakefield

    Eyfield: an array of floats
        Laser field

    Exfield: an array of floats
        Transverse Wakefield

    xlong: 1D numpy array
        longitudinal x values

    xtrans: 1D numpy array
        transverse x values

    gamma_g: float (for now)
        group velocity of the laser

    dt: float
        time step

    zinit: 1-2D numpy array of shape (1,1) or (1,2)
        initial values for position

    pinit: 1-2D numpy array of shape (1,1) or (1,2)
        initial values for moment

    tinit: float
        initial value for time

    num_steps: int
        number of time steps

    latency: int
        the latency for injection for each particle in steps

    num_particles: int
        number of particles

    diag_period:int
        period of array saving

    accel: int
        acceleration for the adaptative time step

    Returns:
    --------
    t_array: a 1-2D array of floats of shape (num_steps, num_particles )
        indicates the time

    p_array: a 1-2D array of floats of shape (num_steps, num_particles )
        indicates the moment

    z_array: a 1-2D array of floats of shape (num_steps, num_particles )
        indicates the position
    """

    #generate position of an electron
    #for part in xrange(num_particles):
    #    randint()
    def f(t, z):
        # Look up the field that corresponds to the position
        xlongdiff = np.absolute( xlong[1]- xlong[0] )
        if dimension == "2D":
            indexlong = int((z[0] - np.min(xlong))/long_dynamics*len(xlong))
            xlongdiff = np.absolute( xlong[indexlong+1]- xlong[indexlong] )
            indextransx = int((z[1] - np.min(xtrans))/trans_dynamics*len(xtrans))
            xtransxdiff = np.absolute( xtrans[indextransx+1]- xtrans[indextransx] )

            Ezpartlong = Ezfield[indextransx, indexlong]*( \
                     xlong[indexlong +1] - z[0])/(xlongdiff) + \
                     Ezfield[indextransx, indexlong + 1]\
                     *(z[0] - xlong[indexlong] )/(xlongdiff)

            Ezparttrans = Ezfield[ indextransx, indexlong ]*( \
                     xtrans[indextransx+1] - z[1])/(xtransxdiff) + \
                     Ezfield[indextransx + 1, indexlong]*( \
                     z[1]- xtrans[indextransx] )/(xtransxdiff)

            Expartlong = Exfield[indextransx, indexlong]*( \
                    xlong[indexlong +1] - z[0])/(xlongdiff) + \
                    Exfield[indextransx, indexlong + 1]\
                    *(z[0] - xlong[indexlong] )/(xlongdiff)

            Exparttrans = Exfield[ indextransx , indexlong ]*( \
                     xtrans[indextransx + 1] - z[1])/(xtransxdiff) + \
                     Exfield[indextransx + 1, indexlong]*( \
                     z[1]- xtrans[indextransx] )/(xtransxdiff)

            E = np.array([(Ezpartlong + Ezparttrans)/2  ,
                        (Expartlong + Exparttrans)/2])
        elif dimension == "1D":

            indexlong = int((np.max(xlong) - z )*len(xlong)/long_dynamics)
            #print indexlong
            if indexlong == 0 or indexlong == np.shape(xlong)[0]:
                Ezpartlong = Ezfield[indexlong]
                Eypartlong = Eyfield[indexlong]
            else:
                Ezpartlong = Ezfield[indexlong]*(z - xlong[indexlong +1]   )\
                        /(xlongdiff) + Ezfield[indexlong+1]*( \
                        xlong[indexlong] - z )/(xlongdiff)

                Eypartlong = Eyfield[indexlong]*(z - xlong[indexlong +1] )\
                        /(xlongdiff) + Eyfield[indexlong+1]*( \
                        xlong[indexlong] - z )/(xlongdiff)

            E = Ezpartlong
        return -e*E

    def g(t, p, gamma_g):
        # first index corresponds to the longitudinal direction
        # 2nd index corresponds to the transverse direction
        if dimension == "1D":

            gamma = np.sqrt(1 + (p**2)/(m_e*c)**2)
            plong = p/(m_e*gamma) - c*(1 - 0.5/gamma_g**2)
            p = plong

        elif dimension == "2D":
            gamma = np.sqrt(1 + (p[0]**2 + p[1]**2)/(m_e*c)**2)
            ptrans = p[1]/(m_e*gamma)
            plong = p[0]/(m_e*gamma) - c*(1 - 0.5/gamma_g**2)
            p = np.array([plong, ptrans])

        return p


    latency_array = []

    if latency is not None:
        k = 0
        for part in xrange(num_particles):
            latency_array.append(k*latency)
            k+=1
    else:
        latency_array = np.tile(0, num_particles)
        t = np.tile(0.0, num_particles)
    #pdb.set_trace()
    if np.shape(zinit)[0]==1:
        z = np.tile(zinit, [num_particles,1])
        p = p_old = np.tile(pinit, [num_particles,1])

    else:
        z = np.array( zinit )
        p = p_old = np.tile(pinit, num_particles)

    z_array = [[] for i in xrange(num_particles)] # for position
    t_array = [[] for i in xrange(num_particles)] # for time
    p_array = [[] for i in xrange(num_particles)] # for moment

    #Initialization
    t = np.tile(tinit, num_particles)
    h = np.tile(dt, num_particles)

    index = 0
    long_dynamics = np.max(xlong)-np.min(xlong)
    h_init = dt

    # Looking for buckets
    index_non_zero = np.compress(np.array(xlong)<=0, np.arange(len(xlong)))
    root_value = findRoot( Ezfield[index_non_zero[0]:-1],
    xlong[index_non_zero[0]:] )

    if xtrans is not None:
        trans_dynamics = np.max(xtrans)-np.min(xtrans)

    if alreadyAccel is None:
        moreThanGammaG = np.zeros(num_particles, dtype=int)
    else:
        moreThanGammaG = np.tile(alreadyAccel, num_particles)

    lessThanGammaG = np.zeros(num_particles, dtype=int)
    #print z

    while index<num_steps:
        if index%(int(num_steps/5))==0:
            print "step: %d" %index

        for part in xrange(num_particles):

            if index >= latency_array[part]:
                try:
                    if z[part]>root_value[1]:
                        # save x, y, z values
                        if index%diag_period ==0:
                            t_array[part].append( t[part].copy() )
                            p_array[part].append( p[part].copy() )
                            z_array[part].append( z[part].copy() )

                        #Runge Kutta method
                        k1 = h[part]*f( t[part], z[part] )
                        l1 = h[part]*g( t[part], p[part], gamma_g )
                        k2 = h[part]*f( t[part] + h[part]/2, z[part] )
                        l2 = h[part]*g( t[part] + h[part]/2, p[part] + k1/2, gamma_g )
                        k3 = h[part]*f( t[part] + h[part]/2, z[part] )
                        l3 = h[part]*g( t[part] + h[part]/2, p[part] + k2/2, gamma_g )
                        k4 = h[part]*f( t[part] + h[part], z[part])
                        l4 = h[part]*g( t[part] + h[part], p[part] + k3, gamma_g )

                        p[part] = p[part] + (1./6)* ( k1 + 2*k2 + 2*k3 + k4 )
                        z[part] = z[part] + (1./6)* ( l1 + 2*l2 + 2*l3 + l4 )

                        #Adaptative time step
                        if dimension =="2D":
                            psmc = p[part][0]/(m_e*c)
                        else:
                            psmc = p[part]/(m_e*c)

                        if psmc >= gamma_g and moreThanGammaG[part] == 0:
                            print "Adapt activated at %d" %(index)
                            h[part] *= accel
                            moreThanGammaG[part] += 1
                        else:
                            h[part] = h_init

                        # incrementation
                        t[part] += h[part]
                        p_old[part] = p[part]
                    #print p[part]
                    #pdb.set_trace()
                except IndexError:
                    #pdb.set_trace()
                    break
        # index incrementation
        index += 1

    return t_array, z_array, p_array
