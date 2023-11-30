A few observations:

1. Numba's gets very slow at JIT-compiling the largest matrix (EQUILIBRIUM RATES JACOBIAN), jumping from 8.7 secs on Python 3.9 to 218.6 secs on Python 3.10. That's 3.6 minutes.
2. JIT-compiling the non-sparsified `arr[i,j] = k` `_def` functions is just as slow in Python 3.8 (271 secs for eq_rates_jac_def) as it is in Python 3.11 (283 secs). Perhaps Numba removes implicit no-ops in `np.array([...])` during compilation in Python 3.8 but not in Python 3.11.
3. Before JIT compilation, running the sparsified `arr[i,j] = k` functions is faster than the `np.array([...])` lambdas by a factor that scales with array size (up to a 10x improvement for equilibrium rates Jacobian).
4. Before JIT compilation, running the non-sparsified `arr[i,j] = k` functions is faster than the `np.array([...])` lambdas for small arrays (non-Jacobians) but slower for large arrays (Jacobians).
5. After JIT compilation, all `arr[i,j] = k` functions (both sparsified `_sp` and non-sparsified) run approximately 4x faster than their `np.array([...])` counterparts (the upper case names, below).
6. There is some internal caching happening in Numba that makes it hard to accurately measure compilation time. For example, tcs_rates, tcs_rates_sp and 2CS RATES all compile in around 850 ms when run in their own Python 3.8 interpreters. However, in the timings above, tcs_rates and 2CS RATES finish compiling in about 550 ms each. Accounting for this, all sparsified `arr[i,j] = k` functions compile more quickly than their `np.array([...])` counterparts.


### Numba timings on Python 3.8.13 on macOS-13.6-x86_64-i386-64bit  
2CS RATES                   0.015978366999999993 ms, jitting 1,184.653908 ms, jitted 0.006510348099999996 ms, iterations to payoff 125,122  
2CS RATES JACOBIAN          0.0953379048 ms, jitting 2,001.2414290000002 ms, jitted 0.011066804699999988 ms, iterations to payoff 23,748  
EQUILIBRIUM RATES           0.03115936140000004 ms, jitting 1,158.9600180000002 ms, jitted 0.008109368399999983 ms, iterations to payoff 50,280  
EQUILIBRIUM RATES JACOBIAN  0.2879765488 ms, jitting 8,901.701184 ms, jitted 0.01672013000000021 ms, iterations to payoff 32,817  

tcs_rates_def               0.010450016099999716 ms, jitting 569.8722420000024 ms, jitted 0.001720164900000043 ms, iterations to payoff 65,279  
tcs_rates_jac_def           0.13356808200000003 ms, jitting 20,005.776208 ms, jitted 0.0021936018999994646 ms, iterations to payoff 152,281  
eq_rates_def                0.029005911000000138 ms, jitting 980.3659860000025 ms, jitted 0.0018896867999998789 ms, iterations to payoff 36,154  
eq_rates_jac_def            0.4050972475000002 ms, jitting 271,254.86866 ms, jitted 0.0033040562999985925 ms, iterations to payoff 675,111  

tcs_rates_sp                0.011431847700004027 ms, jitting 453.82445600000665 ms, jitted 0.0015253669000003358 ms, iterations to payoff 45,811  
tcs_rates_jac_sp            0.01360706030000074 ms, jitting 573.0455499999607 ms, jitted 0.0017255909999960295 ms, iterations to payoff 48,230  
eq_rates_sp                 0.029484997100001922 ms, jitting 1,027.308971000025 ms, jitted 0.0018602831999999126 ms, iterations to payoff 37,188  
eq_rates_jac_sp             0.03182189599999674 ms, jitting 1,394.5140220000098 ms, jitted 0.002677237400001786 ms, iterations to payoff 47,848  

tcs_rates_sp_lambda         0.08006414190000441 ms, jitting 809.5578869999827 ms, jitted 0.0017876297999976033 ms, iterations to payoff 10,342  
tcs_rates_jac_sp_lambda     0.0907719549000035 ms, jitting 1,185.5144409999525 ms, jitted 0.0020915035000030006 ms, iterations to payoff 13,368  
eq_rates_sp_lambda          0.07687059840000075 ms, jitting 1,404.7398189999853 ms, jitted 0.002013057800002116 ms, iterations to payoff 18,766  
eq_rates_jac_sp_lambda      0.2040152667999962 ms, jitting 2,678.165186000001 ms, jitted 0.003008811199998718 ms, iterations to payoff 13,324  


### Numba timings on Python 3.9.0 on macOS-10.16-x86_64-i386-64bit  
2CS RATES                   0.015723539099999996 ms, jitting 990.0004150000001 ms, jitted 0.006475504200000004 ms, iterations to payoff 107,050  
2CS RATES JACOBIAN          0.09399640450000002 ms, jitting 2,019.0535950000003 ms, jitted 0.010718396499999994 ms, iterations to payoff 24,245  
EQUILIBRIUM RATES           0.031742993900000016 ms, jitting 1,149.7406740000001 ms, jitted 0.007975839499999983 ms, iterations to payoff 48,375  
EQUILIBRIUM RATES JACOBIAN  0.28882374890000007 ms, jitting 8,663.410468999999 ms, jitted 0.016904203700000054 ms, iterations to payoff 31,860  

tcs_rates_def               0.010573521999999969 ms, jitting 604.0812279999983 ms, jitted 0.0017162975000001523 ms, iterations to payoff 68,202  
tcs_rates_jac_def           0.13282950940000013 ms, jitting 18,267.870500999998 ms, jitted 0.0022173831000003477 ms, iterations to payoff 139,864  
eq_rates_def                0.026540149699999915 ms, jitting 1,017.7430639999941 ms, jitted 0.0017883490999999196 ms, iterations to payoff 41,118  
eq_rates_jac_def            0.4094844586999997 ms, jitting 273,723.239352 ms, jitted 0.0030730794000021436 ms, iterations to payoff 673,513  

tcs_rates_sp                0.011168816300005345 ms, jitting 438.36830499998314 ms, jitted 0.0016022448000001077 ms, iterations to payoff 45,823  
tcs_rates_jac_sp            0.01376097119999713 ms, jitting 622.3916480000184 ms, jitted 0.001891743799995993 ms, iterations to payoff 52,437  
eq_rates_sp                 0.02771735430000035 ms, jitting 986.862828000028 ms, jitted 0.0017713481999976465 ms, iterations to payoff 38,035  
eq_rates_jac_sp             0.03203025729999922 ms, jitting 1,380.249785999979 ms, jitted 0.0025994564999962224 ms, iterations to payoff 46,898  

tcs_rates_sp_lambda         0.07778004049999936 ms, jitting 786.8671519999566 ms, jitted 0.001765110100001266 ms, iterations to payoff 10,351  
tcs_rates_jac_sp_lambda     0.09074097069999992 ms, jitting 1,193.0652869999676 ms, jitted 0.0020814714999971783 ms, iterations to payoff 13,457  
eq_rates_sp_lambda          0.0745214848000046 ms, jitting 1,350.4579350000085 ms, jitted 0.0019872305999967923 ms, iterations to payoff 18,618  
eq_rates_jac_sp_lambda      0.21307888239999784 ms, jitting 2,681.3003099999833 ms, jitted 0.0030173352000019806 ms, iterations to payoff 12,764  


### Numba timings on Python 3.10.9 on macOS-13.6-x86_64-i386-64bit  
2CS RATES                   0.015975806093774737 ms, jitting 1,129.613304976374 ms, jitted 0.006940389191731811 ms, iterations to payoff 125,021  
2CS RATES JACOBIAN          0.11463893190957607 ms, jitting 26,837.867535068654 ms, jitted 0.02803794969804585 ms, iterations to payoff 309,903  
EQUILIBRIUM RATES           0.03245552959851921 ms, jitting 1,790.798002970405 ms, jitted 0.008799140702467411 ms, iterations to payoff 75,700  
EQUILIBRIUM RATES JACOBIAN  0.3562066875048913 ms, jitting 218,579.63427295908 ms, jitted 0.06490171670448035 ms, iterations to payoff 750,346  

tcs_rates_def               0.01054359630215913 ms, jitting 620.8321510348469 ms, jitted 0.0016829279949888587 ms, iterations to payoff 70,066  
tcs_rates_jac_def           0.12869117480004208 ms, jitting 25,360.610092058778 ms, jitted 0.0025218613096512853 ms, iterations to payoff 201,005  
eq_rates_def                0.02855758200166747 ms, jitting 1,222.3158030537888 ms, jitted 0.0018193597090430556 ms, iterations to payoff 45,714  
eq_rates_jac_def            0.43003707149764525 ms, jitting 302,556.1538609909 ms, jitted 0.0031688884948380294 ms, iterations to payoff 708,781  

tcs_rates_sp                0.01148243179777637 ms, jitting 456.5289040328935 ms, jitted 0.0016454229946248234 ms, iterations to payoff 46,409  
tcs_rates_jac_sp            0.013752656907308846 ms, jitting 587.4081309884787 ms, jitted 0.002044259093236178 ms, iterations to payoff 50,170  
eq_rates_sp                 0.02770394890103489 ms, jitting 1,222.5532369920984 ms, jitted 0.0017992118024267255 ms, iterations to payoff 47,194  
eq_rates_jac_sp             0.03228410860756412 ms, jitting 1,367.1569100115448 ms, jitted 0.002633911895100027 ms, iterations to payoff 46,110  

tcs_rates_sp_lambda         0.08229684239486232 ms, jitting 837.0804849546403 ms, jitted 0.0017519350978545845 ms, iterations to payoff 10,393  
tcs_rates_jac_sp_lambda     0.09691802519373595 ms, jitting 3,018.1215530028567 ms, jitted 0.0019365799031220378 ms, iterations to payoff 31,776  
eq_rates_sp_lambda          0.08069587629288436 ms, jitting 2,174.9310919549316 ms, jitted 0.0019888220005668702 ms, iterations to payoff 27,633  
eq_rates_jac_sp_lambda      0.21569996749749407 ms, jitting 8,684.295262908563 ms, jitted 0.0031198805896565317 ms, iterations to payoff 40,852  


### Numba timings on Python 3.11.6 on macOS-13.6-x86_64-i386-64bit  
2CS RATES                   0.01587814970407635 ms, jitting 947.9425499448553 ms, jitted 0.006601949303876609 ms, iterations to payoff 102,191  
2CS RATES JACOBIAN          0.10557840089313686 ms, jitting 24,919.735249946825 ms, jitted 0.02908360930159688 ms, iterations to payoff 325,770  
EQUILIBRIUM RATES           0.030706309399101888 ms, jitting 1,510.4271629825234 ms, jitted 0.008604182093404232 ms, iterations to payoff 68,339  
EQUILIBRIUM RATES JACOBIAN  0.3279961026040837 ms, jitting 196,437.36396392342 ms, jitted 0.07101052849320695 ms, iterations to payoff 764,391  

tcs_rates_def               0.011116337694693357 ms, jitting 575.7137710461393 ms, jitted 0.0015750429010950029 ms, iterations to payoff 60,339  
tcs_rates_jac_def           0.12754201089264824 ms, jitting 20,596.30252304487 ms, jitted 0.0022405575960874557 ms, iterations to payoff 164,374  
eq_rates_def                0.026441552198957653 ms, jitting 938.6886970605701 ms, jitted 0.001725771394558251 ms, iterations to payoff 37,979  
eq_rates_jac_def            0.4249145888024941 ms, jitting 282,930.52829802036 ms, jitted 0.0029723537038080393 ms, iterations to payoff 670,543  

tcs_rates_sp                0.010805666598025709 ms, jitting 380.95901906490326 ms, jitted 0.001368631306104362 ms, iterations to payoff 40,369  
tcs_rates_jac_sp            0.012970933399628848 ms, jitting 507.30835006106645 ms, jitted 0.0016872612992301585 ms, iterations to payoff 44,960  
eq_rates_sp                 0.02440052960300818 ms, jitting 864.55695098266 ms, jitted 0.0017182315001264214 ms, iterations to payoff 38,116  
eq_rates_jac_sp             0.03133286029333249 ms, jitting 1,159.440755029209 ms, jitted 0.0024173545069061217 ms, iterations to payoff 40,098  

tcs_rates_sp_lambda         0.07688086410053074 ms, jitting 737.6813619630411 ms, jitted 0.001716534502338618 ms, iterations to payoff 9,814  
tcs_rates_jac_sp_lambda     0.09333062359364704 ms, jitting 2,539.8723609978333 ms, jitted 0.0020686439936980603 ms, iterations to payoff 27,831  
eq_rates_sp_lambda          0.08049982450902461 ms, jitting 1,900.499640032649 ms, jitted 0.0018513152026571333 ms, iterations to payoff 24,164  
eq_rates_jac_sp_lambda      0.21978946550516412 ms, jitting 7,492.27405898273 ms, jitted 0.0025912325945682824 ms, iterations to payoff 34,495  
