import pandas
from dotenv import load_dotenv
import os
import warnings
import matplotlib
from Manage import Manager

if __name__ == '__main__':
    load_dotenv()
    NEPTUNE_API_TOKEN = os.getenv('NEPTUNE-API-TOKEN')
    ALPHA_VANTAGE_TOKEN = os.getenv('ALPHA-VANTAGE-API-TOKEN')

    # disables unnecessary warnings
    pandas.options.mode.chained_assignment = None  # default='warn'
    matplotlib.use('SVG')
    warnings.filterwarnings(action='ignore', category=UserWarning)

    manager = Manager(NEPTUNE_API_TOKEN, ALPHA_VANTAGE_TOKEN)
    manager.trade_for_day()
