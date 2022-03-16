'''Implementation of AC coding method
'''

__all__ = ['ac_code_of']

# PCPNS: Physicochemical property names
PCPNS = ['H1', 'H2', 'NCI', 'P1', 'P2', 'SASA', 'V']

# AAPCPVS: Physicochemical property values of amino acid
AAPCPVS = {
    'A': { 'H1': 0.62, 'H2':-0.5, 'NCI': 0.007187, 'P1': 8.1, 'P2':0.046, 'SASA':1.181, 'V': 27.5 },
    'C': { 'H1': 0.29, 'H2':-1.0, 'NCI':-0.036610, 'P1': 5.5, 'P2':0.128, 'SASA':1.461, 'V': 44.6 },
    'D': { 'H1':-0.90, 'H2': 3.0, 'NCI':-0.023820, 'P1':13.0, 'P2':0.105, 'SASA':1.587, 'V': 40.0 },
    'E': { 'H1': 0.74, 'H2': 3.0, 'NCI': 0.006802, 'P1':12.3, 'P2':0.151, 'SASA':1.862, 'V': 62.0 },
    'F': { 'H1': 1.19, 'H2':-2.5, 'NCI': 0.037552, 'P1': 5.2, 'P2':0.290, 'SASA':2.228, 'V':115.5 },
    'G': { 'H1': 0.48, 'H2': 0.0, 'NCI': 0.179052, 'P1': 9.0, 'P2':0.000, 'SASA':0.881, 'V':  0.0 },
    'H': { 'H1':-0.40, 'H2':-0.5, 'NCI':-0.010690, 'P1':10.4, 'P2':0.230, 'SASA':2.025, 'V': 79.0 },
    'I': { 'H1': 1.38, 'H2':-1.8, 'NCI': 0.021631, 'P1': 5.2, 'P2':0.186, 'SASA':1.810, 'V': 93.5 },
    'K': { 'H1':-1.50, 'H2': 3.0, 'NCI': 0.017708, 'P1':11.3, 'P2':0.219, 'SASA':2.258, 'V':100.0 },
    'L': { 'H1': 1.06, 'H2':-1.8, 'NCI': 0.051672, 'P1': 4.9, 'P2':0.186, 'SASA':1.931, 'V': 93.5 },
    'M': { 'H1': 0.64, 'H2':-1.3, 'NCI': 0.002683, 'P1': 5.7, 'P2':0.221, 'SASA':2.034, 'V': 94.1 },
    'N': { 'H1':-0.78, 'H2': 2.0, 'NCI': 0.005392, 'P1':11.6, 'P2':0.134, 'SASA':1.655, 'V': 58.7 },
    'P': { 'H1': 0.12, 'H2': 0.0, 'NCI': 0.239531, 'P1': 8.0, 'P2':0.131, 'SASA':1.468, 'V': 41.9 },
    'Q': { 'H1':-0.85, 'H2': 0.2, 'NCI': 0.049211, 'P1':10.5, 'P2':0.180, 'SASA':1.932, 'V': 80.7 },
    'R': { 'H1':-2.53, 'H2': 3.0, 'NCI': 0.043587, 'P1':10.5, 'P2':0.291, 'SASA':2.560, 'V':105.0 },
    'S': { 'H1':-0.18, 'H2': 0.3, 'NCI': 0.004627, 'P1': 9.2, 'P2':0.062, 'SASA':1.298, 'V': 29.3 },
    'T': { 'H1':-0.05, 'H2':-0.4, 'NCI': 0.003352, 'P1': 8.6, 'P2':0.108, 'SASA':1.525, 'V': 51.3 },
    'V': { 'H1': 1.08, 'H2':-1.5, 'NCI': 0.057004, 'P1': 5.9, 'P2':0.140, 'SASA':1.645, 'V': 71.5 },
    'W': { 'H1': 0.81, 'H2':-3.4, 'NCI': 0.037977, 'P1': 5.4, 'P2':0.409, 'SASA':2.663, 'V':145.5 },
    'Y': { 'H1': 0.26, 'H2':-2.3, 'NCI': 117.3000, 'P1': 6.2, 'P2':0.298, 'SASA':2.368, 'V':  0.023599 },
}

import math

def avg_sd(NUMBERS):
    AVG = sum(NUMBERS)/len(NUMBERS)
    TEM = [pow(NUMBER-AVG, 2) for NUMBER in NUMBERS]
    DEV = sum(TEM)/len(TEM)
    SD = math.sqrt(DEV)
    return (AVG, SD)

# PCPVS: Physicochemical property values
PCPVS = {'H1':[], 'H2':[], 'NCI':[], 'P1':[], 'P2':[], 'SASA':[], 'V':[]}
for AA, PCPS in AAPCPVS.items():
    for PCPN in PCPNS:
        PCPVS[PCPN].append(PCPS[PCPN])

# PCPASDS: Physicochemical property avg and sds
PCPASDS = {}
for PCP, VS in PCPVS.items():
    PCPASDS[PCP] = avg_sd(VS)

# NORMALIZED_AAPCPVS
NORMALIZED_AAPCPVS = {}
for AA, PCPS in AAPCPVS.items():
    NORMALIZED_PCPVS = {}
    for PCP, V in PCPS.items():
        NORMALIZED_PCPVS[PCP] = (V-PCPASDS[PCP][0])/PCPASDS[PCP][1]
    NORMALIZED_AAPCPVS[AA] = NORMALIZED_PCPVS

def pcp_value_of(AA, PCP):
    """Get physicochemical properties value of amino acid."""
    return NORMALIZED_AAPCPVS[AA][PCP];

def pcp_sequence_of(PS, PCP):
    """Make physicochemical properties sequence of protein sequence."""
    PCPS = []
    for I, CH in enumerate(PS):
        PCPS.append(pcp_value_of(CH, PCP))
    # Centralization
    AVG = sum(PCPS)/len(PCPS)
    for I, PCP in enumerate(PCPS):
        PCPS[I] = PCP - AVG
    return PCPS

def ac_values_of(PS, PCP, LAG):
    """Get ac values of protein sequence."""
    AVS = []
    PCPS = pcp_sequence_of(PS, PCP)
    for LG in range(1, LAG+1):
        SUM = 0
        for I in range(len(PCPS)-LG):
            SUM = SUM + PCPS[I]*PCPS[I+LG]
        SUM = SUM / (len(PCPS)-LG)
        AVS.append(SUM)
    return AVS

def all_ac_values_of(PS, LAG):
    """Get all ac values of protein sequence."""
    AAVS = []
    for PCP in PCPS:
        AVS = ac_values_of(PS, PCP, LAG)
        AAVS = AAVS + AVS
    return AAVS

def ac_code_of(PS):
    """Get ac code of protein sequence."""
    AC_Code = all_ac_values_of(PS, 30)
    # Normalizing AC_Code
    # MIN_CODE = min(AC_Code)
    # MAX_CODE = max(AC_Code)
    # AC_Code = [(N-MIN_CODE)*1.0/(MAX_CODE-MIN_CODE) for N in AC_Code]
    return AC_Code

if __name__=="__main__":
    AC = ac_code_of('MKFVYKEEHPFEKRRSEGEKIRKKYPDRVPVIVEKAPKARIGDLDKKKYLVPSDLTVGQFYFLIRKRIHLRAEDALFFFVNNVIPPTSATMGQLYQEHHEEDFFLYIAYSDESVYGL')
    print(AC)
