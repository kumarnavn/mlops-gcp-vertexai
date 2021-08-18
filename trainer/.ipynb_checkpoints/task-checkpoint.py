from collections import defaultdict
import warnings
import pandas as pd
warnings.filterwarnings("ignore")
import pandas as pd
import argparse
import hypertune
from greykite.framework.templates.autogen.forecast_config import ForecastConfig
from greykite.framework.templates.autogen.forecast_config import MetadataParam
from greykite.framework.templates.forecaster import Forecaster 
from greykite.framework.templates.model_templates import ModelTemplateEnum
from greykite.framework.utils.result_summary import summarize_grid_search_results
from greykite.framework.templates.autogen.forecast_config import ModelComponentsParam
from greykite.common.evaluation import EvaluationMetricEnum
from greykite.framework.templates.autogen.forecast_config import EvaluationMetricParam
from greykite.framework.templates.autogen.forecast_config import EvaluationPeriodParam

def resample(df):
    df = df.copy()    
    df['ts'] =  pd.to_datetime(df['ts'])
    max_ts =  df['ts'].max()
    min_ts =  df['ts'].min()
    dates = pd.DataFrame(pd.date_range(min_ts, max_ts, freq='W'), columns = ['ts'])
    df = dates.merge(df, on =['ts'], how='left')
    df.fillna(0, inplace=True)
    return df



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    # Input Arguments
    
    parser.add_argument(
        'filename',        
         type = str        
    )

    parser.add_argument(
        'input_folder',           
         type = str        
    )


    args = parser.parse_args()
    arguments = args.__dict__

    input_gcs  =  arguments['input_folder'] + '/'+ arguments['filename']
    output_gcs =  arguments['input_folder']+'/'+'output/'+arguments['filename']
    df =  pd.read_csv(input_gcs)
    df['WEEK_END_DATE'] = df['WEEK_END_DATE'].astype('datetime64[ns]')
    df.rename(columns = {'WEEK_END_DATE': 'ts', 'CLEASNSED_PLUS_OPEN_CUST_REQUESTED_QTY_CS': 'y'}, inplace = True)
    df['ts'] =  pd.to_datetime(df['ts'])
    df['y'] =  df['y'].astype('float')
    df['ts'] = (pd.to_datetime(df['ts'])+ pd.to_timedelta (1,'d')).dt.date

    df = resample(df)

    custom_model_components = ModelComponentsParam(
    custom={
        "fit_algorithm_dict": {
            "fit_algorithm": "ridge",
            "fit_algorithm_params": None
        }
    }
    )

    metadata = MetadataParam(
    time_col='ts',  # name of the time column
    value_col='y',  # name of the value column
    freq="W"  #"MS" for Montly at start date, "H" for hourly, "D" for daily, "W" for weekly, etc.
    )
    forecaster = Forecaster()
    result = forecaster.run_forecast_config(
    df=df,
    config=ForecastConfig(
        model_template=ModelTemplateEnum.SILVERKITE.name,
        model_components_param=custom_model_components,
        forecast_horizon=6,  # forecasts 100 steps ahead
        coverage=0.95,  # 95% prediction intervals
        metadata_param=metadata
    )
    )

    forecast = result.forecast

    final_df =  forecast.df

    final_df.to_csv(output_gcs)

    grid_search = result.grid_search
    cv_results = summarize_grid_search_results(
    grid_search=grid_search,
    decimals=2,
    # The below saves space in the printed output. Remove to show all available metrics and columns.
    cv_report_metrics=None,
    column_order=["rank", "mean_test", "split_test", "mean_train", "split_train", "mean_fit_time", "mean_score_time", "params"])
    # Transposes to save space in the printed output
    cv_results["params"] = cv_results["params"].astype(str)
    cv_results.set_index("params", drop=True, inplace=True)
    #cv_results = cv_results.transpose()
    cv_mape = cv_results['mean_test_MAPE'].values[0]

    hpt = hypertune.HyperTune()
    hpt.report_hyperparameter_tuning_metric(
    hyperparameter_metric_tag='cv_mean_mape',
    metric_value=cv_mape)

        

