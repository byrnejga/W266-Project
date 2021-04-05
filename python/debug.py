# Constants for use in messages
DEBUG = False
DETAIL = False

PRIOR_DEBUG = False
PRIOR_DETAIL = False

# Returns setting to prior debug state.
# If nothing will change, it will exit without a message0
def last():
    global DEBUG, DETAIL, PRIOR_DEBUG, PRIOR_DETAIL
    if DEBUG != PRIOR_DEBUG or DETAIL != PRIOR_DETAIL:

        DEBUG, PRIOR_DEBUG = PRIOR_DEBUG, DEBUG
        DETAIL, PRIOR_DETAIL = PRIOR_DETAIL, DETAIL

        if DETAIL == True:
            print("*************** DEBUG DETAILS TURNED ON *****************")
        elif DEBUG == True:
            print("*************** DEBUG TURNED ON *****************")
        else:
            print("***************** DEBUG TURNED OFF **********************")
    

def off():
    global DEBUG, DETAIL, PRIOR_DEBUG, PRIOR_DETAIL
    DEBUG, PRIOR_DEBUG = False, DEBUG
    DETAIL, PRIOR_DETAIL = False, DETAIL
    print("***************** DEBUG TURNED OFF **********************")

def on():
    global DEBUG, DETAIL, PRIOR_DEBUG, PRIOR_DETAIL
    DEBUG, PRIOR_DEBUG = True, DEBUG
    DETAIL, PRIOR_DETAIL = False, DETAIL
    print("****************** DEBUG TURNED ON **********************")

def show_detail():
    global DEBUG, DETAIL, PRIOR_DEBUG, PRIOR_DETAIL
    DEBUG, PRIOR_DEBUG = True, DEBUG
    DETAIL, PRIOR_DETAIL = True, DETAIL
    print("*************** DEBUG DETAILS TURNED ON *****************")


def msg(*arg):
    global DEBUG
    if DEBUG:
        print(*arg)

def detail(*arg):
    global DETAIL
    if DETAIL:
        print(*arg)

