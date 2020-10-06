# %%
import sys
#sys.path.append('/home/users/altonr')
import rngStream
#from scipy.stats import norm  # normal distribution for CIs
import pandas as pd
import numpy as np
import time
import math
#import random

# NOTES:
# sex 1 := male; sex 2:= female
__author__ = "Alton Russell"


# %% Simulation Model
class Sim_Model:
    """ Instantiates simulation model """

    # %% Initialization
    # for both PSA and non-PSA; US and PR

    #@profile
    def __init__(self, PSA):
        t=int( time.time() * 1000.0 )
        self.start = time.time()
        self.timeString = ""
        # random.seed(a=123)
        # Random number generator      
        self.PSA = PSA
        # PARAMETERS
        # Survival
        # CDC life table p_death
        self.p_death_healthy = [0.005831063, 0.000370078, 0.000248410,
                                0.000183835, 0.000158093, 0.000141899,
                                0.000125960, 0.000112999, 0.000101496,
                                0.000092268, 0.000088595, 0.000095882,
                                0.000120441, 0.000165984, 0.000228521,
                                0.000295707, 0.000365616, 0.000445652,
                                0.000533718, 0.000622837, 0.000713076,
                                0.000795011, 0.000857717, 0.000897847,
                                0.000922493, 0.000943054, 0.000967305,
                                0.000993542, 0.001023970, 0.001057757,
                                0.001093930, 0.001131248, 0.001170190,
                                0.001211833, 0.001259149, 0.001317635,
                                0.001388096, 0.001466288, 0.001549088,
                                0.001638092, 0.001741038, 0.001861899,
                                0.001998998, 0.002157166, 0.002341890,
                                0.002543562, 0.002773737, 0.003054293,
                                0.003385186, 0.003746207, 0.004109361,
                                0.004473667, 0.004861702, 0.005287972,
                                0.005754040, 0.006254293, 0.006769531,
                                0.007294195, 0.007823044, 0.008368036,
                                0.008958804, 0.009605531, 0.010287694,
                                0.010995333, 0.011735077, 0.012519374,
                                0.013393533, 0.014396520, 0.015578928,
                                0.017000714, 0.018678572, 0.020548509,
                                0.022558674, 0.024701593, 0.026983518,
                                0.029456904, 0.032338053, 0.035667565,
                                0.039459255, 0.043922257, 0.048756212,
                                0.053978011, 0.059674047, 0.066394895,
                                0.073868088, 0.081965879, 0.091544263,
                                0.102046624, 0.113516033, 0.125986904,
                                0.139482349, 0.154011607, 0.169567525,
                                0.186124220, 0.203635275, 0.222032622,
                                0.241226137, 0.261104524, 0.281536996,
                                0.302376240, 1.000000000]
        # multiplier derived from SCANDAT by bin
        self.surv_mult = pd.Series([-0.817572418110032, -0.813276839365234,
                                    -0.797255332490873, -0.796849824185898, -0.79389313639905, -0.788511714568704,
                                    -0.785958671736801, -0.778819817626416, -0.776996855799508, -0.77468002220649,
                                    -0.773511707057567, -0.770957286490019, -0.768091531084629, -0.767861689953426,
                                    -0.764936741259688, -0.761923124147346, -0.760928033300991, -0.757204208586802,
                                    -0.752917080591849, -0.752746349586526, -0.75140021756903, -0.750034076821807,
                                    -0.747413823684727, -0.743547158650365, -0.737859557401385, -0.737720943842025,
                                    -0.737275764221944, -0.735683527739676, -0.735319502027239, -0.734885190074792,
                                    -0.733364801161838, -0.732552162235254, -0.731571583485507, -0.7299628395368,
                                    -0.727997197668908, -0.727705820084743, -0.725136499691371, -0.724432636786181,
                                    -0.723955979443069, -0.721608410206325, -0.720420333352485, -0.719413966075051,
                                    -0.714501341533066, -0.711670127108074, -0.710399485143846, -0.709618768639444,
                                    -0.709577753842831, -0.709515845538552, -0.708520965144798, -0.707221322118684,
                                    -0.705628462903537, -0.704279072938498, -0.699977726376569, -0.693087484256031,
                                    -0.691340408861523, -0.681419540343925, -0.679104793883406, -0.670283295124512,
                                    -0.661069074272555, -0.657565790561358, -0.646130514189894, -0.642647996239707,
                                    -0.633325515629018, -0.631280742765057, -0.623842152698512, -0.596598212512847,
                                    -0.594285903987529, -0.582534174204996, -0.581110787220343, -0.514117833707892,
                                    -0.48637077676596, -0.449511020731904, -0.368944671501742, -0.366154771453697,
                                    -0.292639044770105, -0.280746675672264, -0.153440011414076, -0.032664173435345,
                                    0.518510271429626, 1.4651623712012, 1.89076717961945],
                                   [3331, 3231, 3233, 3133, 3333, 3321, 3123, 3332, 3132, 3131,
                                    3323, 3232, 2133, 3223, 2123, 2332, 2333, 2331, 3213, 3313,
                                    2321, 2233, 2323, 2232, 2231, 1233, 2223, 2132, 2313, 1333,
                                    2213, 3221, 1332, 3113, 1331, 1123, 1321, 2131, 3322, 1223,
                                    1323, 2322, 1132, 1133, 1232, 2111, 1313, 1322, 1131, 1231,
                                    1213, 3111, 2113, 2221, 1122, 1222, 1221, 1113, 3222, 2222,
                                    3121, 3122, 1312, 2312, 2122, 3312, 2121, 1121, 1111, 2212,
                                    1212, 3212, 1311, 1112, 2112, 3112, 2311, 3311, 3211, 2211,
                                    1211])
        self.surv_mult_max = max(self.surv_mult)
        #  Kleinman unadjusted
        self.surv_unadj = [[0.895, 0.877, 0.862, 0.862, 0.848],
                           [0.729, 0.667, 0.630, 0.601, 0.572],
                           [0.618, 0.507, 0.435, 0.400, 0.312]]
        # Productivity
        self.prodByAge = pd.Series([0, 17659, 39069, 46840, 56235, 60254, 
                                    60426, 60795, 57858, 52843, 44921, 41798,
                                    31279],
                                    [0, 15, 25, 30, 35, 40, 45, 50, 55, 60, 65, 
                                     70, 75])
        
        # Consumption
        self.consumpByAge = pd.Series([17219, 17219, 19569.6296296296, 19569.6296296296, 
                                    19542.3529411765, 19542.3529411765, 25416.4285714286, 
                                    25416.4285714286, 27884.5454545455, 27884.5454545455, 
                                    28262.7777777778, 28262.7777777778, 24181.875],
                                    [0, 15, 25, 30, 35, 40, 45, 50, 55, 60, 65, 
                                     70, 75])
                                     
        self.prev_vect = np.geomspace(start = 1e-8, stop = 1e-2, num = 13)

        # RUN-SPECIFIC VARIABLES
        # recipient characteristics
        self.age = 0
        self.sex = 0
        self.unitsRBC = 0
        self.unitsFFP = 0
        self.unitsPLT = 0
        ##load scandat
        self.SCANDATdata = pd.read_csv('SCANDATA.csv')
        #print(self.SCANDATdata)
        
        self.setSeed(t)
        
        #Variables that do not vary in PSA
        # Discount rate
        self.rdisc = .03

        if PSA == 0:
            # probabiity parameters
            self.p_flu_like_symptoms = 0.1836
            self.p_GBS = 0.000257
            self.p_GBS_death = .0258
            self.p_GBS_perm = pd.Series([0, .0484, .121, .22, .488],
                                        index=[0, 15, 20, 35, 65])
            self.p_sexual_transmission = pd.Series([0, .00161, .021423, .05386, .07749,
                                                    .07343, .06316, .04509, .04016, .02820],
                                                   index=[0, 10, 15, 20, 25, 30, 40, 50, 60, 70])
            self.p_CZS = .0343
            self.p_stillborn_CZS = .07
            # self.separateInventoryCompliance = 1

            # Cost parameters
            self.costTP = self.costFP = 85  # cost of true positive/false positive
            self.costFluR = 1257.24
            self.costFluP = 100.97
            self.costGBS = 57107  # in first year
            self.costGBS_perm = 35721.83  # per year
            self.costGBS_death = 67107
            self.costBirth = 23505.9
            self.costInfantZikvTest = 224.8
            self.costMotherZikvTest = 467.6
            self.costStillBirth = 6152.6
            self.costZAM_birth = 24196.1
            self.costZAM_lifetime = 4035892.6
            self.separateInventoryPerUnitCost = 25  # does not include ID-NAT costs
            
            # utility/QALY/disease-state duration parameters
            self.uBaseline = 0.9
            self.uFlu = pd.DataFrame(pd.DataFrame({1: pd.Series([.57, .58, .63, .61, .59],
                                                                index=[0, 20, 35, 50, 65]),
                                                   2: pd.Series([.5, .59, .58, .55, .54],
                                                                index=[0, 20, 35, 50,
                                                                       65])}))  # indexed by recipient age
            self.p_recip_pregnant_by_age = pd.Series ([0,
                                                       0.023357573,
                                                       0.05734304,
                                                       0.062300081,
                                                       0.054130878,
                                                       0.030337086,
                                                       0.007693326,
                                                       0], 
                                                      index=[0, 15, 20, 25, 30, 35, 40, 45])
    
            self.p_partner_pregnant_by_age = pd.Series ([0, 0.044175, 0.10845, 
                                                         0.117825, 0.102375, 
                                                         0.057375, 0.01455, 
                                                         0], index=[0, 15, 20, 
                                                          25, 30, 35, 40, 45])
            self.uGBS_yr1 = .76
            self.uGBS_yr2 = .87
            self.uGBS_yr3to6 = 0.99
            self.uFlu_partner = .43
            self.durationFlu = 21/(365*1.0)
            self.durationFlu_partner = 7/(365*1.0)
            self.CZS_QALYloss = 79.8

            self.transmissability_RBC = 0.5
            self.transmissability_PLT = 0.5
            self.transmissability_FFP = 0.9

            self.tests = np.array([[0, 0, 1], [10, .99, .99997], [6, 0.925, .999999999]])
        else:  # PSA variable sampling
            self.PSA_init()
        
    def PSA_init(self):
        # probabiity parameters
        self.p_flu_like_symptoms = np.random.beta(15.22805, 66.67302)
        self.p_GBS = np.random.beta(42, 163357)
        self.p_GBS_death = np.random.beta(128, 4954)
        self.p_GBS_perm = pd.Series([0, 
                                     .4*np.random.beta(187.7445, 1362.2845), 
                                     np.random.beta(187.7445, 1362.2845), 
                                     np.random.beta(1007.739, 3572.178), 
                                     np.random.beta(1534.888, 1610.365)],
                                    index=[0, 15, 20, 35, 65])
        self.p_trans_given_sex = np.random.triangular(.01, .09, .2)
        self.p_sexual_transmission = pd.Series([0, 
                                                .0161*self.p_trans_given_sex, 
                                                .21423*self.p_trans_given_sex, 
                                                .5386*self.p_trans_given_sex, 
                                                .7749*self.p_trans_given_sex,
                                                .7343*self.p_trans_given_sex, 
                                                .6316*self.p_trans_given_sex, 
                                                .4509*self.p_trans_given_sex, 
                                                .4016*self.p_trans_given_sex, 
                                                .2820*self.p_trans_given_sex],
                                               index=[0, 10, 15, 20, 25, 30, 40, 50, 60, 70])
        self.p_CZS = np.random.beta(3.611713, 92.806642)
        self.p_stillborn_CZS = np.random.triangular(.054, .07, .084)
        # self.separateInventoryCompliance = 1

        # Cost parameters
        self.costTP = self.costFP = np.random.triangular(70, 85, 100)  # cost of true positive/false positive
        self.costFluR = np.random.lognormal(7.136697, 0.306482)
        self.costFluP = np.random.lognormal(4.6146026, 0.2137651)
        self.costGBS = np.random.lognormal(10.95222344, 0.08618574)  # in first year, permanent or temporary
        self.costGBS_perm = np.random.triangular(35721.83*.8, 35721.83, 35721.83*1.2)  # per year>1, permanent
        self.costGBS_death = np.random.triangular(67107*.8, 67107, 67107*1.2)
        self.costBirth = np.random.triangular(18803, 23505.9, 28205.6)
        self.costInfantZikvTest = np.random.triangular(180, 224.8, 270)
        self.costMotherZikvTest = np.random.triangular(373.9, 467.6, 561.4)
        self.costStillBirth = np.random.triangular(4922.3, 6152.6, 7382.9)
        self.costZAM_birth = np.random.triangular(19356.9, 24196.1, 29035.3)
        self.costZAM_lifetime = np.random.triangular(2389385.8, 3811038.5, 5907253.5)
        self.separateInventoryPerUnitCost = np.random.triangular(10, 25, 40)  # does not include ID-NAT costs
        
        # utility/QALY/disease-state duration parameters
        self.uBaseline = 0.9
        self.uFlu = pd.DataFrame(pd.DataFrame({1: pd.Series([
                np.random.beta(11.25936, 11.25937), 
                np.random.beta(15.10341, 10.58873), 
                np.random.beta(28.93211, 21.04579), 
                np.random.beta(27.98735, 22.96893), 
                np.random.beta(16.96525, 14.49232)],
                index=[0, 20, 35, 50, 65]),
                                               2: pd.Series([
                np.random.beta(17.16510, 13.00793), 
                np.random.beta(26.94199, 19.58631), 
                np.random.beta(44.15273, 26.03280), 
                np.random.beta(32.33860, 20.77625), 
                np.random.beta(10.52540, 7.39798)],
                index=[0, 20, 35, 50,65])}))  # indexed by recipient age
        
        self.p_recip_preg = np.random.triangular(0.0001319, 0.001993, 0.0025749)
        self.p_recip_pregnant_by_age = pd.Series ([
                0,
                self.p_recip_preg*9.071254495,
                self.p_recip_preg*22.27000679,
                self.p_recip_preg*24.19514569,
                self.p_recip_preg*21.02251678,
                self.p_recip_preg*11.78185007,
                self.p_recip_preg*2.987815572,
                0
                ], index=[0, 15, 20, 25, 30, 35, 40, 45])

        self.p_partner_pregnant_by_age = pd.Series ([0, 0.044175, 0.10845, 
                                                     0.117825, 0.102375, 
                                                     0.057375, 0.01455, 
                                                     0], index=[0, 15, 20, 
                                                      25, 30, 35, 40, 45])
        self.uGBS_yr1 = np.random.beta(3.873550, 1.140319)
        self.uGBS_yr2 = np.random.beta(1.8029145, 0.2907072)
        self.uGBS_yr3to6 = 0.99
        self.uFlu_partner = np.random.beta(449.3405, 595.5460)
        self.durationFlu = np.random.lognormal(2.9874569, 0.9569858)/(365*1.0)
        self.durationFlu_partner = np.random.lognormal(1.9784724, 0.4483528)/(365*1.0)
        self.CZS_QALYloss = 79.8

        spec_ID = np.random.beta(358763, 9)
        self.tests = np.array([[0, 0, 1], #baseline cost, sens, spec
                      [np.random.triangular(7, 10, 13), #ID-NAT cost
                       np.random.beta(198, 2), #sens
                       spec_ID], #spec
                       [np.random.triangular(3, 6, 9), #MP-NAT cost
                        np.random.beta(185, 15), #sens
                        1-(1-spec_ID)**2]]) #spec
        self.transmissability_RBC = np.random.triangular(0.3, 0.5, 0.7)
        self.transmissability_PLT = np.random.triangular(0.3, 0.5, 0.7)
        self.transmissability_FFP = np.random.triangular(0.8, 0.9, 1.0)   
        
    #@profile
    def zeroOutcomes(self):
        self.TTZ = [0] * 3  # transfusion transmission of ZIKV
        self.costRecip = 0  # cost accrued related to recipient ailments
        self.costInfant = 0  # cost accrued related to infant
        self.costPartner = 0  # cost accrued related to sexual partner
        self.costPartnerInfant = 0
        self.costProductivity = 0
        self.QALYLrecip = 0  # QALY accrued in rep
        self.QALYLinfant = 0
        self.QALYLpartner = 0
        self.QALYLpartnerInfant = 0
        self.CZS = 0  # congenital zika syndromes
        self.GBS = 0  # Guillain-barre syndrome cases
        self.fluLike = 0  # flu-like symptoms of recipient
        self.probTTZ = np.empty([len(self.prev_vect)])
        self.survival = 0

    def setSeed(self, t):
        t1= ( ((t & 0xff000000) >> 24) +
             ((t & 0x00ff0000) >>  8) +
             ((t & 0x0000ff00) <<  8) +
             ((t & 0x000000ff) << 24))
        
        t=int( time.time() * 1000.0 )
        t2= (((t & 0xff000000) >> 24) +
             ((t & 0x00ff0000) >>  8) +
             ((t & 0x0000ff00) <<  8) +
             ((t & 0x000000ff) << 24))
        
        d = ([(t1//(10**i))%10 for i in range(int(math.ceil(math.log(t1, 10)))-1, -1, -1)] +
                  [(t2//(10**i))%10 for i in range(int(math.ceil(math.log(t2, 10)))-1, -1, -1)] + [0, 0, 0])
        seed = [int(d[0]*1000+d[1]*100+d[2]*10+d[3]),
                int(d[19]*1000+d[4]*100+d[5]*10+d[6]),
                int(d[7]*100+d[8]*10+d[9]),
                int(d[10]*100+d[11]*10+d[12]),
                int(d[13]*100+d[14]*10+d[15]),
                int(d[16]*100+d[17]*10+d[18])]
        print(seed)
        rngStream.SetPackageSeed(seed)
        self.unigen = rngStream.RngStream("coin flip")
        self.unigen.ResetNextSubstream()
    # %% Utilities
    # Discount factor, survival, distributions for PSA
    
    #@profile
    def discFac(self, t1, t2):
        return (1 - np.exp(-1 * self.rdisc * (t2 - t1))) / (self.rdisc * (t2 - t1))
    
    #@profile
    def getSurvival(self):
        # Convert age, units RBC, FFP, PLT into bins to pull in survival multiplier
        if self.age < 45:
            agebin = 1
        elif self.age < 65:
            agebin = 2
        else:
            agebin = 3

        if self.unitsRBC == 0:
            RBCbin = 1
        elif self.unitsRBC < 5:
            RBCbin = 2
        else:
            RBCbin = 3

        if self.unitsPLT == 0:
            PLTbin = 1
        elif self.unitsPLT < 5:
            PLTbin = 2
        else:
            PLTbin = 3

        if self.unitsFFP == 0:
            FFPbin = 1
        elif self.unitsFFP < 5:
            FFPbin = 2
        else:
            FFPbin = 3
            

        lookup = agebin * 1000 + RBCbin * 100 + PLTbin * 10 + FFPbin
        mult = self.surv_mult[lookup]
        # print("lookup, mult", lookup, mult)
        # Generate survival cumprob
        n = 100 - self.age
        cumSurv = np.zeros(int(n + 1))
        healthySurv = 1 - self.p_death_healthy[self.age]
        surv_unadj = self.surv_unadj[agebin - 1][0]
        cumSurv[0] = surv_unadj + (mult / self.surv_mult_max) * (healthySurv - surv_unadj)

        # print("healthy, unadj, cumSurv[0]", healthySurv, surv_unadj, cumSurv[0])

        # Will first fill cumSurv with the probability of surviving to a year
        for i in range(1, 5):
            healthySurv = healthySurv * (1 - self.p_death_healthy[self.age + i])
            surv_unadj = self.surv_unadj[agebin - 1][i]
            cumSurv[i] = surv_unadj + (mult / self.surv_mult_max) * (healthySurv - surv_unadj)
        for i in range(5, n):
            cumSurv[i] = cumSurv[i - 1] * (1 - self.p_death_healthy[self.age + i])

        # Then do '1-' in order to give the probability of dying in a given year
        cumSurv = 1 - cumSurv

        self.survival = next(x[0] for x in enumerate(cumSurv)
                             if x[1] > self.unigen.RandU01()) + self.unigen.RandU01()
        self.bl_QALY = self.survival * self.uBaseline * self.discFac(0, self.survival)
        # print("cumSurv", cumSurv)
        # print("survival", self.survival)
        
    #@profile
    def addProdCost(self, duration, ageStart, mort):
        #print("duration", duration)
        #If mortality, first do the consumption
        cLoss = 0 #tallies consumption loss
        if mort==1:
            yearsAdded = 0   #tracks years of productivity that have been added
            cDuration = duration #duration for adding consumption
            while cDuration > 0:
                currentBracket = self.consumpByAge.index[self.consumpByAge.index<=ageStart+yearsAdded].max()
                nextBracket = self.consumpByAge.index[self.consumpByAge.index>ageStart+yearsAdded].min()
                if ageStart + cDuration > nextBracket:
                    timeLost = nextBracket - currentBracket
                else:
                    timeLost = cDuration
                yearsAdded += timeLost
                cLoss += timeLost*self.consumpByAge[currentBracket]*self.discFac(max(currentBracket, ageStart) - ageStart, yearsAdded)
                cDuration = cDuration - timeLost
            #print("Consumption lost", cLoss)
            
        
        yearsAdded = 0   #tracks years of productivity that ahve been added
        prodLoss = 0 #tracks productivity loss
        while duration > 0:
            currentBracket = self.prodByAge.index[self.prodByAge.index<=ageStart+yearsAdded].max()
            nextBracket = self.prodByAge.index[self.prodByAge.index>ageStart+yearsAdded].min()
            if ageStart + duration > nextBracket:
                timeLost = nextBracket - currentBracket
            else:
                timeLost = duration
            yearsAdded += timeLost
            prodLoss += timeLost*self.prodByAge[currentBracket]*self.discFac(max(currentBracket, ageStart) - ageStart, yearsAdded)
            duration = duration - timeLost

            #print("current", currentBracket, "next", nextBracket,"timeLost", timeLost, "prodLoss", prodLoss)
        #print("ageStart", ageStart, "Mort", mort, "ProdLoss", prodLoss, "cLoss", cLoss)
        return prodLoss - cLoss

    #@profile
    def isPregnant(self, age, sex, recip):
        if self.unigen.RandU01() < self.getPPregnant(age, sex, recip):
            return 1
        return 0
    
    def getPPregnant(self, age, sex, recip):
        p_pregnant=0
        if sex == 2:
            if recip == 1:
                p_pregnant = self.p_recip_pregnant_by_age.get_value(
                    self.p_recip_pregnant_by_age.index.asof(self.age))
            else:
                p_pregnant = self.p_partner_pregnant_by_age.get_value(
                    self.p_partner_pregnant_by_age.index.asof(self.age))
            #print("Sex", sex, "Age", age, "p_pregnant", p_pregnant)
        return p_pregnant


    # %% SAMPLING RECIPIENT and TESTING FOR ZIKA
    
    def getRecip(self):
        recipIndex = int(math.floor(self.unigen.RandU01() * 790824))
        #print(self.SCANDATdata.iloc[[recipIndex]])
        self.age = int(self.SCANDATdata.get_value(recipIndex, 4, takeable=True) + math.floor(self.unigen.RandU01() * 5))
        # print("age", self.age)
        self.sex = self.SCANDATdata.get_value(recipIndex, 0, takeable=True)
        self.unitsFFP = self.SCANDATdata.get_value(recipIndex, 3, takeable=True)*0.408765293
        self.unitsRBC = self.SCANDATdata.get_value(recipIndex, 1, takeable=True)*0.408765293
        self.unitsPLT = self.SCANDATdata.get_value(recipIndex, 2, takeable=True)*0.408765293
        
        #print("Age, RBC, PLT, FFP", self.age, self.unitsRBC, self.unitsPLT, self.unitsFFP)
        self.getSurvival()
        #print("Survival", self.survival)

    #@profile
    def recipZikaRisk(self):

        # Zika positive units calculations.
        probNegUnit = 1-self.prev_vect
        
        self.probTTZ = (1 - 
                        (1 - self.transmissability_RBC*(1-probNegUnit**self.unitsRBC))*
                        (1 - self.transmissability_PLT*(1-probNegUnit**self.unitsPLT))*
                        (1 - self.transmissability_FFP*(1-probNegUnit**self.unitsFFP)))
        # print("COSTS", self.costBC, "PROB.TTZ", self.probTTZ)


    # %% RECIPIENT OUTCOMES
    # Recipient, sexual partner, infants to recipient or sexual partner

    #@profile
    def recipOutcomes(self):
        randRecip = self.unigen.RandU01();
        fluLike = 0;
        GBS = 0
        costProd = 0
        # check recipiant outcomes: flu-like symptoms, GBS, or asymptomatic
        # flu-like symptoms
        if randRecip < self.p_flu_like_symptoms:
            fluLike = 1
            costRecip = self.costFluR
            uFactor = self.uFlu.get_value(self.uFlu.index.asof(self.age), self.sex)
            QALYLrecip = min(self.survival, self.durationFlu) * self.uBaseline * (1-uFactor) * self.discFac(0, min(
                self.survival, self.durationFlu))
            costProd = self.addProdCost(min(self.survival, self.durationFlu), self.age, 0)
            #print("RF", QALYLrecip, costRecip, costProd)

        # GBS
        elif randRecip < (self.p_flu_like_symptoms + self.p_GBS):
            GBS = 1
            randGBS = self.unigen.RandU01()
            if randGBS < self.p_GBS_death:  # GBS death
                costRecip = self.costGBS_death
                QALYLrecip = self.bl_QALY
                costProd = self.addProdCost(self.survival, self.age, 1)
                #print("RGD", QALYLrecip, costRecip, costProd)
            elif randGBS < self.p_GBS_death + self.p_GBS_perm.get_value(
                    self.p_GBS_perm.index.asof(self.age)):  # GBS permanent disability
                costRecip = self.costGBS_perm * self.survival * self.discFac(0, self.survival)
                QALYLrecip = (self.survival * self.uGBS_yr1 * 
                              self.uBaseline * self.discFac(0, self.survival))
                costProd = self.addProdCost(self.survival, self.age, 0)
                #print("RGP", QALYLrecip, costRecip, costProd)
            else:  # GBS temporary symptoms
                costRecip = self.costGBS
                QALYLrecip = (self.uGBS_yr1 * self.uBaseline *
                                    min(self.survival, 1) * 
                                    self.discFac(0,min(self.survival, 1)))
                # print("Yr1 QALY:",QALYLrecip)
                if self.survival > 1:
                    QALYLrecip += (self.uGBS_yr2 * self.uBaseline *
                                   (min(self.survival, 2) - 1) * 
                                   self.discFac(1, min(self.survival, 2)))
                    # print("Yr2 QALY:",QALYLrecip)
                if self.survival > 2:
                    QALYLrecip += (self.uGBS_yr3to6 * self.uBaseline *
                                   (min(self.survival, 6) - 2) * 
                                   self.discFac(2, min(self.survival, 6)))
                    # print("Yr6 QALY:",QALYLrecip)
                costProd = self.addProdCost(min(self.survival, 2), self.age, 0)
                #print("RGT", QALYLrecip, costRecip, costProd)

        else: #Asymptomatic
            costRecip = 0
            QALYLrecip = 0
        # if GBS>0:
        # print("Recipient outcomes. Cost:", costRecip, "QALY:", QALYLrecip,
        #      "BLQALY:",self.uBaseline*self.survival*self.discFac(0,self.survival),
        #      "flulike:", fluLike, "GBS:",GBS)
        return costRecip, costProd, QALYLrecip, fluLike, GBS

    #@profile
    def infantOutcomes(self):
        costInfant = self.costInfantZikvTest + self.costMotherZikvTest
        QALYLinfant = 0
        CZS = 0
        costProd = 0
        randCZS = self.unigen.RandU01()
        if randCZS < self.p_CZS:  # Congenital ZIKV syndrome
            QALYLinfant = self.CZS_QALYloss * self.discFac(0, self.CZS_QALYloss)
            costProd = self.addProdCost(79.8, 0, 1)
            CZS = 1
            randCZS_outcome = self.unigen.RandU01()
            if randCZS_outcome < self.p_stillborn_CZS:  # Stillborn
                costInfant = self.costStillBirth - self.costBirth
                #print("StillbornCost", costInfant)
                #print("IST", QALYLinfant, costInfant, costProd)
            else:  # Live birth
                costInfant = (self.costZAM_birth - self.costBirth) + self.costZAM_lifetime
                #print("LiveBirthCost", costInfant)
                # print("infant outcomes. Cost:",costInfant, "uLOSS:", QALYLinfant,"CZS:", CZS)
                #print("ILB", QALYLinfant, costInfant, costProd)
        return costInfant, costProd, QALYLinfant, CZS

    #@profile
    def partnerOutcomes(self):
        costPartner = 0
        costProd = 0
        QALYLpartner = 0
        costPartnerInfant = 0
        QALYLpartnerInfant = 0
        partnerCZS = 0
        GBS = 0
        # Check for transfusion transmission of ZIKV to sexual partner
        randTransmit = self.unigen.RandU01()
        #sexual transmission
        if randTransmit < self.p_sexual_transmission.get_value(self.p_sexual_transmission.index.asof(self.age)):
            randPartner = self.unigen.RandU01()
            # check recipient outcomes: flu-like symptoms, GBS, or asymptomatic
            # flu-like symptoms
            if randPartner < self.p_flu_like_symptoms:
                costPartner = self.costFluP
                QALYLpartner = (1 - self.uFlu_partner)*self.durationFlu_partner*self.discFac(0, self.durationFlu_partner)
                costProd = self.addProdCost(self.durationFlu_partner, self.age, 0)
                #print("PF", QALYLpartner, costPartner, costProd)
                
            # GBS
            elif randPartner < (self.p_flu_like_symptoms + self.p_GBS):
                GBS = 1
                randGBS = self.unigen.RandU01()
                partnerLifeExp = max(2, 78.74-self.age)
                if randGBS < self.p_GBS_death:  # GBS death
                    costPartner = self.costGBS_death
                    QALYLpartner = partnerLifeExp
                    costProd = self.addProdCost(partnerLifeExp, self.age, 1)
                    #print("PGD", QALYLpartner, costPartner, costProd)
                elif randGBS < self.p_GBS_death + self.p_GBS_perm.get_value(
                        self.p_GBS_perm.index.asof(self.age)):  # GBS permanent disability
                    costPartner = self.costGBS_perm * partnerLifeExp * self.discFac(0, partnerLifeExp)
                    QALYLpartner = self.uGBS_yr1 * partnerLifeExp * self.discFac(0, partnerLifeExp)

                    
                    costProd = self.addProdCost(partnerLifeExp, self.age, 0)
                    #print("PGP", QALYLpartner, costPartner, costProd)
                else:  # GBS temporary symptoms
                    costPartner = self.costGBS
                    QALYLpartner = (self.uGBS_yr1 *
                                    min(partnerLifeExp, 1) * 
                                    self.discFac(0,min(partnerLifeExp, 1)))
                    # print("Yr1 QALY:",QALYLrecip)
                    if partnerLifeExp > 1:
                        QALYLpartner += (self.uGBS_yr2 * 
                                         (min(partnerLifeExp, 2) - 1) * 
                                         self.discFac(1, min(partnerLifeExp, 2)))
                        # print("Yr2 QALY:",QALYLrecip)
                    if partnerLifeExp > 2:
                        QALYLpartner += (self.uGBS_yr3to6 * 
                                         (min(partnerLifeExp, 6) - 2) * 
                                         self.discFac(2, min(partnerLifeExp, 6)))
                        # print("Yr6 QALY:",QALYLrecip)
                    costProd = self.addProdCost(min(partnerLifeExp, 2), self.age, 0)
                    #print("PGT", QALYLpartner, costPartner, costProd)

            else:
                costPartner = 0
                QALYLpartner = 0
            pregnant = self.isPregnant(self.age, 2, 0)  # see if sexual partner pregnant; assume same age as recip
            if pregnant == 1:  # if pregnant add infant cost/qaly to partner cost/utility
                costPartnerInfant, costProdInf, QALYLpartnerInfant, partnerCZS = self.infantOutcomes()
                costProd += costProdInf
        # print("Partner outcomes. Cost:",costPartner,"Uloss:",QALYLpartner,"costPartnerInfant:",
        #      costPartnerInfant, "UlossPartnerInfant:", QALYLpartnerInfant, "partnerCZS:", partnerCZS)
        return costPartner, costProd, QALYLpartner, costPartnerInfant, QALYLpartnerInfant, partnerCZS, GBS

    #@profile
    def outcomes(self, pregnant):  
        CZS = 0;
        costPartner = 0;
        QALYLpartner = 0;
        costPartnerInfant = 0;
        QALYLpartnerInfant = 0
        costInfant = 0;
        QALYLinfant = 0
        # Determine outcomes only for scenarios that had a case of ZIKV transmission
        # Get recipient outcomes
        costRecip, costProd, QALYLrecip, fluLike, GBS = self.recipOutcomes()
        # If recipient male get sexual partner outcomes
        if self.sex == 1:
            costPartner, costProdPartner, QALYLpartner, costPartnerInfant, QALYLpartnerInfant, CZS, GBS_partner = self.partnerOutcomes()
            costProd += costProdPartner
            GBS += GBS_partner
        # If recipient pregnant get infant outcomes
        if pregnant == 1:
            costInfant, costProdInfant, QALYLinfant, CZS = self.infantOutcomes()
            costProd += costProdInfant
        # Then assign costs/QALY losses for recipient/partners/infants/partnerInfants by policy
        self.costRecip = costRecip
        self.costInfant = costInfant
        self.costPartner = costPartner
        self.costPartnerInfant = costPartnerInfant
        self.QALYLrecip = QALYLrecip
        self.QALYLinfant = QALYLinfant
        self.QALYLpartner = QALYLpartner
        self.QALYLpartnerInfant = QALYLpartnerInfant
        self.CZS = CZS
        self.GBS = GBS
        self.fluLike = fluLike
        self.costProductivity = costProd
        
    #@profile
    def getColHeaders(self, PSA):
        cols = []
        if PSA == 1:
            cols = cols + ["p_flu_like_symptoms",
                           "p_GBS", 
                           "p_GBS_death",
                           "p_GBS_perm_15to19",
                           "p_GBS_perm_20to34",
                           "p_GBS_perm_35to64",
                           "p_GBS_perm_65up",
                           "p_trans_given_sex",
                           "p_CZS",
                           "p_stillborn_CZS",
                           "costFluR",
                           "costFluP",
                           "costGBS",
                           "costGBS_perm",
                           "costGBS_death",
                           "costBirth",
                           "costInfantZikvTest",
                           "costMotherZikvTest",
                           "costStillBirth",
                           "costZAM_birth",
                           "costZAM_lifetime",
                           "uFlu_male_0to19",
                           "uFlu_male_20to34",
                           "uFlu_male_35to49",
                           "uFlu_male_50to64",
                           "uFlu_male_65up",
                           "uFlu_female_0to19",
                           "uFlu_female_20to34",
                           "uFlu_female_35to49",
                           "uFlu_female_50to64",
                           "uFlu_female_65up",
                           "p_recip_preg",
                           "uGBS_yr1",
                           "uGBS_yr2",
                           "uFlu_partner",
                           "durationFlu",
                           "durationFlu_partner",
                           "transmiss_RBC",
                           "transmiss_PLT",
                           "transmiss_FFP"]
            
        cols = cols + ['unitsRBC', 'unitsPLT', 'unitsFFP', 'Baseline_LY', 'Baseline_QALY']
        cols = cols + ["probTTZ" + str(i) for i in range(len(self.prev_vect))]
        cols = cols + ["costRecip" + str(i) for i in range(len(self.prev_vect))]
        cols = cols + ["costInfant" + str(i) for i in range(len(self.prev_vect))]
        cols = cols + ["costPartner" + str(i) for i in range(len(self.prev_vect))]
        cols = cols + ["costPartnerInfant" + str(i) for i in range(len(self.prev_vect))]
        cols = cols + ["costProductivity" + str(i) for i in range(len(self.prev_vect))]
        cols = cols + ["QALYLrecip" + str(i) for i in range(len(self.prev_vect))]
        cols = cols + ["QALYLinfant" + str(i) for i in range(len(self.prev_vect))]
        cols = cols + ["QALYLpartner" + str(i) for i in range(len(self.prev_vect))]
        cols = cols + ["QALYLpartnerInfant" + str(i) for i in range(len(self.prev_vect))]
        cols = cols + ["CZS" + str(i) for i in range(len(self.prev_vect))]
        cols = cols + ["GBS" + str(i) for i in range(len(self.prev_vect))]
        cols = cols + ["flulike" + str(i) for i in range(len(self.prev_vect))]
        return cols

    def get_PSA_row(self):
        row = np.array([self.p_flu_like_symptoms,
                            self.p_GBS, 
                            self.p_GBS_death,
                            self.p_GBS_perm.get(15),
                            self.p_GBS_perm.get(20),
                            self.p_GBS_perm.get(35),
                            self.p_GBS_perm.get(65),
                            self.p_trans_given_sex,
                            self.p_CZS, 
                            self.p_stillborn_CZS,
                            self.costFluR,
                            self.costFluP,
                            self.costGBS,
                            self.costGBS_perm,
                            self.costGBS_death,
                            self.costBirth,
                            self.costInfantZikvTest,
                            self.costMotherZikvTest,
                            self.costStillBirth,
                            self.costZAM_birth,
                            self.costZAM_lifetime,
                            self.uFlu.get_value(0,1),
                            self.uFlu.get_value(20,1),
                            self.uFlu.get_value(35,1),
                            self.uFlu.get_value(50,1),
                            self.uFlu.get_value(65,1),
                            self.uFlu.get_value(0,2),
                            self.uFlu.get_value(20,2),
                            self.uFlu.get_value(35,2),
                            self.uFlu.get_value(50,2),
                            self.uFlu.get_value(65,2),
                            self.p_recip_preg,
                            self.uGBS_yr1,
                            self.uGBS_yr2,
                            self.uFlu_partner,
                            self.durationFlu,
                            self.durationFlu_partner,
                            self.transmissability_RBC,
                            self.transmissability_PLT,
                            self.transmissability_FFP])
        return row
        
    #@profile
    def getRow(self):
        row = np.concatenate(( 
                np.array([self.unitsRBC, self.unitsPLT, self.unitsFFP, 
                          self.survival, self.bl_QALY]), 
                self.probTTZ,
                np.array(self.costRecip*self.probTTZ), 
                np.array(self.costInfant*self.probTTZ), 
                np.array(self.costPartner*self.probTTZ),
                np.array(self.costPartnerInfant*self.probTTZ), 
                np.array(self.costProductivity*self.probTTZ),
                np.array(self.QALYLrecip*self.probTTZ), 
                np.array(self.QALYLinfant*self.probTTZ),
                np.array(self.QALYLpartner*self.probTTZ), 
                np.array(self.QALYLpartnerInfant*self.probTTZ),
                np.array(self.CZS*self.probTTZ), 
                np.array(self.GBS*self.probTTZ), 
                np.array(self.fluLike*self.probTTZ)))
        return row
    
               
    

    # %% RUN SIMULATION

    #@profile
    def doRep(self):# Runs simulation for one recipient

        #Get next recipient
        self.zeroOutcomes()
        self.getRecip()
        self.recipZikaRisk()

        # Get prob pregnant and if separate inventory needed
        p_pregnant = 0
        if self.sex == 2 and self.age >= 15 and self.age < 45:
            p_pregnant = self.getPPregnant(self.age, self.sex, 1)


        self.outcomes(0)
        if p_pregnant == 0:
            row = sim.getRow()
        else:
            row = sim.getRow()*(1-p_pregnant)
            self.outcomes(1)
            #print("PPreg", p_pregnant, "Prob TTZ", self.probTTZ[0], "CZS", self.CZS)
            row = row + sim.getRow()*p_pregnant

        return row

    #@profile
    def runIter(self, PSA):
        # First generate columns specific to the output type and
        # whether it is a PSA run. If outputType0, then we will
        # generate only one row of output, in which we sum all outcomes.
        # If outputType1 (detail for transfusion-transmissions only) or
        # if outputType2 (detail for every recipient), will add rows to the output

        #get columns for output


        rows = np.zeros(174)
        for rep in range(5000000):
            rows = rows + sim.doRep()

        #Get PSA distributions into ouput
        if PSA==1:
            #GET PSA ROW and add to row
            PSA_row = self.get_PSA_row()
            rows = np.concatenate((PSA_row, rows), axis=0)
            
        output = pd.DataFrame(data=[rows], columns=self.headers)
        return output


    #@profile
    def runModel(self, n, PSA):
        # Run one iteration. If n=1, return the output. If n>1, add
        # a column 'Iter' that indexes the outputs and then return the
        # output from all iterations in a large data table.
        self.headers = self.getColHeaders(PSA)
        self.cols = len(self.headers)
        #print("Cols", self.cols, "HEADS", self.headers)

        output = self.runIter(PSA)
        if n==1:
            result = output
        else:
            output['Iter'] = 0
            result = output
            #print("Completed Iter", 0)
            self.timeString += "Iter 0 time: " + str(time.time()-self.start) + "."
            for i in range(1,n):
                if PSA==1:
                    self.PSA_init()
                output = self.runIter(PSA)
                output['Iter'] = i
                #print("Output", output)
                result = pd.concat([result, output])
                #print("Completed Iter", i)
                self.timeString += "Iter "+str(i)+"complete time: " + str(time.time()-self.start) + ". "
                print(self.timeString)
                #print("result", result)
        return result, self.timeString



# %%
if __name__ == "__main__":
    outputFileName = sys.argv[1]
    n = int(sys.argv[2]) #number of iterations to run
    #prev = float(sys.argv[3])
    PSA = 1

    sim = Sim_Model(PSA)  # instantiate simulation model
    output, timeString = sim.runModel(n, PSA)
    print(timeString)
    output.to_csv(outputFileName + ".csv", sep=',')
