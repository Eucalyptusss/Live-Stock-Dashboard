

#Imports
from dotenv import load_dotenv
import os
load_dotenv()

import dash
import dash_core_components as dcc
from dash.dependencies import Input, Output, State
import dash_html_components as html
import dash_bootstrap_components as dbc
import dash_daq as daq
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
apikey = os.getenv("API_KEY")
import fmpsdk
from sklearn.ensemble import GradientBoostingRegressor
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')
from dash_extensions.enrich import Output, DashProxy, Input, MultiplexerTransform
sbs = False
from sklearn.metrics import mean_squared_error

import warnings
warnings.filterwarnings('ignore')

###################################################################################
############################ Functions ############################################
###################################################################################

def get_current_price(ticker):
    cp_dict = fmpsdk.quote(apikey, ticker)
    cp = cp_dict[0]['price']
    return cp
def vader_sentiment_analysis(ticker, limit=10):
    """
    IN: 

    ticker: This is the stock symbol as or trade name. e.g. TSLA, AAPL, GOOG

    limit: This number references how far back in date you want to go. If you
    want to find the date range please use get_timeframe() on this method's returned df.
    
    OUT: Pandas Dataframe
    """
    #Initialize the sentiment analyzer
    sid = SentimentIntensityAnalyzer()
    #Collect the data from the fmpsdk api
    news = fmpsdk.stock_news(apikey=apikey, tickers=ticker, limit=limit)
    #Construct the dataframe with predtermined columns
    df = pd.DataFrame(columns=['Date', 'Title', 'Text', 'CompoundScoreTitle',
                          'PositiveScoreTitle', 'NegativeScoreTitle', 
                          'CompoundScoreText', 'NeutralScoreTitle',
                           'NeutralScoreText', 'Text Sentiment', 'Title Sentiment',
                          'PositiveScoreText', 'NegativeScoreText',
                          'CompoundAvg', 'Symbol'])
    
    dates = []
    titles = []
    text_ = []
    symbols = []


    #Loops through the news variable which contains the data from the api. For 
    #loop the lists dates, titles, & text_ get appended so that the df above can
    #then have three columns with data
    for article in news:
        dates.append(article['publishedDate'])
        titles.append(article['title'])
        text_.append(article['text']) 
        symbols.append(article['symbol'])
    df.Date = pd.to_datetime(dates)
    df.Title = titles
    df.Text = text_
    df.Symbol = symbols
    
    #Removes None values from text that can appear
    df.Text = df.Text.apply(lambda txt: txt if txt != None else('Blank'))
    #Scores the title of the articles using compound score
    df.CompoundScoreTitle = df.Title.apply(lambda title:
    sid.polarity_scores(title)['compound'])
    #Scores the title of the articles using positive score
    df.PositiveScoreTitle = df.Title.apply(lambda title:
        sid.polarity_scores(title)['pos'])
    #Scores the title of the articles using negative score
    df.NegativeScoreTitle = df.Title.apply(lambda title:
        sid.polarity_scores(title)['neg'])
    #Scores the title of the articles using neutral score
    df.NeutralScoreTitle = df.Title.apply(lambda title:
        sid.polarity_scores(title)['neu'])
    #Scores the text of the articles using compound score
    df.CompoundScoreText = df.Text.apply(lambda text:
        sid.polarity_scores(text)['compound'])
    #Scores the text of the articles using positive score
    df.PositiveScoreText = df.Text.apply(lambda text:
        sid.polarity_scores(text)['pos'])
    #Scores the text of the articles using negative score
    df.NegativeScoreText = df.Text.apply(lambda text:
        sid.polarity_scores(text)['neg'])
    #Scores the text of the articles using neutral score
    df.NeutralScoreText = df.Text.apply(lambda text:
        sid.polarity_scores(text)['neu'])
    
    #I will now add two columns to the dataframe that will tell me which way the
    #sentiment analyzer is leaning in regards to tile and text. I will be using
    #the comound score here because it is the best representation of positve/neutral
    #& negative scores.

    #If the title comound score is less than 0 I determine that the title leans negative
    #if the comound score is greater than 0 I dettermine that the title leans
    #positive. Otherwise neutral.
    df['Title Sentiment'] = df.CompoundScoreTitle.apply(lambda score:
                    'Negative' if score < 0 
                     else('Positive' if score > 0
                          else('Neutral')))
                        
    df['Text Sentiment'] = df.CompoundScoreText.apply(lambda score:
                        'Negative' if score < 0 
                         else('Positive' if score > 0
                              else('Neutral')))
    #Now I will add the day of the week
    #Days of the week are numberd 0-6 where 0 is Monday
    df['weekday'] = df.Date.apply(lambda date:
                                  date.weekday())
    #The time corresponding with the date will also need to in it's own column
    #for later processing
    df['time'] = df.Date.apply(lambda date:
                               date.time())
    #Finally I add the average of the comound scores for the title and text of a 
    #given article
    df.CompoundAvg = (df.CompoundScoreTitle + df.CompoundScoreText)/2
    
    df = df[['Date', 'Title', 'Text', 'Text Sentiment', 'Title Sentiment',]]
    table = dbc.Table.from_dataframe(
        df, striped=True, bordered=True, hover=True
    )
    return table

def vader_sentiment_analysis_prediction(apikey, ticker, limit=5000):
    """
    IN: 
    apikey: This should be your personalized api key. You can get one here 
    (https://fmpcloud.io/)

    ticker: This is the stock symbol as or trade name. e.g. TSLA, AAPL, GOOG

    limit: This number references how far back in date you want to go. If you
    want to find the date range please use get_timeframe() on this method's returned df.
    
    OUT: Pandas Dataframe
    """
    #Initialize the sentiment analyzer
    sid = SentimentIntensityAnalyzer()
    #Collect the data from the fmpsdk api
    news = fmpsdk.stock_news(apikey=apikey, tickers=ticker, limit=limit)
    #Construct the dataframe with predtermined columns
    df = pd.DataFrame(columns=['Date', 'Title', 'Text', 'CompoundScoreTitle',
                          'PositiveScoreTitle', 'NegativeScoreTitle', 
                          'CompoundScoreText', 'NeutralScoreTitle',
                           'NeutralScoreText', 'textLeans', 'titleLeans',
                          'PositiveScoreText', 'NegativeScoreText',
                          'CompoundAvg', 'Symbol'])
    
    dates = []
    titles = []
    text_ = []
    symbols = []


    #Loops through the news variable which contains the data from the api. For 
    #loop the lists dates, titles, & text_ get appended so that the df above can
    #then have three columns with data
    for article in news:
        dates.append(article['publishedDate'])
        titles.append(article['title'])
        text_.append(article['text']) 
        symbols.append(article['symbol'])
    df.Date = pd.to_datetime(dates)
    df.Title = titles
    df.Text = text_
    df.Symbol = symbols
    
    #Removes None values from text that can appear
    df.Text = df.Text.apply(lambda txt: txt if txt != None else('Blank'))
    #Scores the title of the articles using compound score
    df.CompoundScoreTitle = df.Title.apply(lambda title:
    sid.polarity_scores(title)['compound'])
    #Scores the title of the articles using positive score
    df.PositiveScoreTitle = df.Title.apply(lambda title:
        sid.polarity_scores(title)['pos'])
    #Scores the title of the articles using negative score
    df.NegativeScoreTitle = df.Title.apply(lambda title:
        sid.polarity_scores(title)['neg'])
    #Scores the title of the articles using neutral score
    df.NeutralScoreTitle = df.Title.apply(lambda title:
        sid.polarity_scores(title)['neu'])
    #Scores the text of the articles using compound score
    df.CompoundScoreText = df.Text.apply(lambda text:
        sid.polarity_scores(text)['compound'])
    #Scores the text of the articles using positive score
    df.PositiveScoreText = df.Text.apply(lambda text:
        sid.polarity_scores(text)['pos'])
    #Scores the text of the articles using negative score
    df.NegativeScoreText = df.Text.apply(lambda text:
        sid.polarity_scores(text)['neg'])
    #Scores the text of the articles using neutral score
    df.NeutralScoreText = df.Text.apply(lambda text:
        sid.polarity_scores(text)['neu'])
    
    #I will now add two columns to the dataframe that will tell me which way the
    #sentiment analyzer is leaning in regards to tile and text. I will be using
    #the comound score here because it is the best representation of positve/neutral
    #& negative scores.

    #If the title comound score is less than 0 I determine that the title leans negative
    #if the comound score is greater than 0 I dettermine that the title leans
    #positive. Otherwise neutral.
    df.titleLeans = df.CompoundScoreTitle.apply(lambda score:
                    'Negative' if score < 0 
                     else('Positive' if score > 0
                          else('Neutral')))
                        
    df.textLeans = df.CompoundScoreText.apply(lambda score:
                        'Negative' if score < 0 
                         else('Positive' if score > 0
                              else('Neutral')))
    #Now I will add the day of the week
    #Days of the week are numberd 0-6 where 0 is Monday
    df['weekday'] = df.Date.apply(lambda date:
                                  date.weekday())
    #The time corresponding with the date will also need to in it's own column
    #for later processing
    df['time'] = df.Date.apply(lambda date:
                               date.time())
    #Finally I add the average of the comound scores for the title and text of a 
    #given article
    df.CompoundAvg = (df.CompoundScoreTitle + df.CompoundScoreText)/2
    
    return df

def build_news_table(stock, quantity):
        news = fmpsdk.stock_news(apikey, stock, limit=quantity)
        news_df = pd.DataFrame(columns=['Publish Date', 'Title', 'Text', 'Site'])
        dates = []
        titles = []
        texts = []
        sites = []
        for article in news:
            dates.append(article['publishedDate'])
            titles.append(article['title'])
            texts.append(article['text'])
            sites.append(article['site'])
            
        news_df['Publish Date'] = dates
        news_df.Title = titles
        news_df.Text = texts
        news_df.Site = sites
        table = dbc.Table.from_dataframe(
            news_df, striped=True, bordered=True, hover=True
        )
        return table

def update_stock_sym(new_sym):
    global stock_sym
    stock_sym = new_sym
def update_sbs(val):
    global sbs
    sbs = val

def trim_news_dates(stock_prediction_day, df):
  """
  This function will trim a provided dataframe to only contain dates relevant to
  the stock_prediction_day.

  IN: 
   - stock_prediction_day: This is the date that one would like to predict a 
   stocks gain/loss should be a string in the form 'YYYY-MM-DD'
   - df: This is the data frame that will be trimmed and returned only contaning
   dates relavant to the prediction
  """

  #Set start_cutoff to one day before stock_prediction_day at 4:00 PM as
  #a pandas datetime object
  end_cutoff = pd.to_datetime(stock_prediction_day + ' 16:00:00')
  start_cutoff = end_cutoff - timedelta(days=1)
  
  return df.loc[(df.Date >= start_cutoff) &
       (df.Date <= end_cutoff)]

def determine_date_range(df):
  """
  IN: pandas data frame that you would like to know the date range of
  OUT: Date range in pandas.datetime format
  """
  end_date = df.Date[0]
  start_date = df.Date[df.shape[0]-1]
  return (start_date, end_date)


def build_insider_table(stock, quantity=10):
    shareholder_activity = fmpsdk.insider_trading(apikey, stock, limit=quantity)
    df = pd.DataFrame(columns=['Filing Date', 'Transaction Date', 'Transaction Type',
                             'Securities Owned', 'Reporting Name', 'Reporting Position',
                             'Securities Transacted', 'Price', 'Security Name'])
    dates = []
    trans_dates = []
    types = []
    securites = []
    names = []
    positions = []
    transacted = []
    prices = []
    security_names = []
    for sa in shareholder_activity:
        try:
            dates.append(sa['filingDate'])
            trans_dates.append(sa['transactionDate'])
            types.append(sa['acquistionOrDisposition'])
            securites.append(sa['securitiesOwned'])
            names.append(sa['reportingName'])
            positions.append(sa['typeOfOwner'])
            transacted.append(sa['securitiesTransacted'])
            prices.append(sa['price'])
            security_names.append(sa['securityName'])
        except:
            continue
        

    df['Filing Date'] = dates
    df['Transaction Date'] = trans_dates
    df['Transaction Type'] = types
    df['Transaction Type'] = df['Transaction Type'].apply(lambda letter:
                                                         'Sale' if letter =='D'
                                                         else('Buy'))
    df['Securities Owned'] = securites
    df['Reporting Name'] = names
    df['Reporting Position'] = positions
    df['Securities Transacted'] = transacted
    df['Price'] = prices
    df['Security Name'] = security_names

    table = dbc.Table.from_dataframe(
        df, striped=True, bordered=True, hover=True
    )
    return table

        
def build_ratings_table(stock):
    ratings = fmpsdk.rating(apikey, stock)
 

    table_header = [
        html.Thead(html.Tr([
            html.Th("Date:"), 
            html.Th(ratings[0]['date'])
            ]))
    ]
    row1 = html.Tr([
        html.Td("Rating Score:"),
        html.Td(ratings[0]['ratingScore'])
    ])
    row2 = html.Tr([
        html.Td("Rating Recommendation:"),
        html.Td(ratings[0]['ratingRecommendation'])
    ])
    row3 = html.Tr([
        html.Td("Return On Equity Score:"),
        html.Td(ratings[0]['ratingDetailsROEScore'])
    ])
    row4 = html.Tr([
        html.Td("Return On Equity Recommendation:"),
        html.Td(ratings[0]['ratingDetailsROERecommendation'])
    ])
    row5 = html.Tr([
        html.Td("Discounted Cash Flow Score:"),
        html.Td(ratings[0]['ratingDetailsDCFScore'])
    ])
    row6 = html.Tr([
        html.Td("Discounted Cash Flow Recommendation:"),
        html.Td(ratings[0]['ratingDetailsDCFRecommendation'])
    ])
    row7 = html.Tr([
        html.Td("Return On Assets Score:"),
        html.Td(ratings[0]['ratingDetailsROAScore'])
    ])
    row8 = html.Tr([
        html.Td("Return On Assets Recommendation:"),
        html.Td(ratings[0]['ratingDetailsROARecommendation'])
    ])
    row9 = html.Tr([
        html.Td("Debt Equity Score:"),
        html.Td(ratings[0]['ratingDetailsDEScore'])
    ])
    row10 = html.Tr([
        html.Td("Debt Equity Recommendation:"),
        html.Td(ratings[0]['ratingDetailsDERecommendation'])
    ])
    row11 = html.Tr([
        html.Td("Price to Earning Score:"),
        html.Td(ratings[0]['ratingDetailsPEScore'])
    ])
    row12 = html.Tr([
        html.Td("Price to Earnings Recommendation:"),
        html.Td(ratings[0]['ratingDetailsPERecommendation'])
    ])
    row13 = html.Tr([
        html.Td("Price-To-Book  Score:"),
        html.Td(ratings[0]['ratingDetailsPBScore'])
    ])
    row14 = html.Tr([
        html.Td("Price-To-Book Recommendation:"),
        html.Td(ratings[0]['ratingDetailsPBRecommendation'])
    ])
    table_body = [html.Tbody([row1, row2, row3, row4, row5, row6, row7, row8, row9, row10,
                                row11, row12, row13, row14])]
    table = dbc.Table(table_header + table_body, bordered=True, 
        hover=True, responsive=True,)

    return table

def get_annual_revenue_df(stock):
    earning_dict = fmpsdk.income_statement(apikey, stock)
    if len(earning_dict) == 0:
        return 0
    dates = []
    revenues = []
    symbols = []
    net_income = []
    new_df = pd.DataFrame(columns=['Dates', 'Revenue', 'Symbol', 'NetIncome' ])
    for element in earning_dict:
        symbols.append(element['symbol'])
        dates.append(element['date'])
        revenues.append(element['revenue'])
        net_income.append(element['netIncome'])


    new_df.Dates = dates
    new_df.Dates = pd.to_datetime(new_df.Dates)
    new_df.Dates = new_df.Dates.apply(lambda date: date.year)
    new_df.Revenue = revenues
    new_df.NetIncome = net_income
    new_df.Revenue = new_df.Revenue.astype('float64')
    new_df.Symbol = symbols
    
    return new_df

def build_eps_df(stock):
    earning_dict = fmpsdk.income_statement(apikey, stock)
    epss = []
    dates = []
    this_df = pd.DataFrame(columns = ['Date', 'EarningPerShare'])
    for element in earning_dict:
        dates.append(element['date'])
        epss.append(element['eps'])

    this_df.Date = dates
    this_df.EarningPerShare = epss
    this_df.Date = pd.to_datetime(this_df.Date)  
    this_df.Date = this_df.Date.apply(lambda date: date.year)
    return this_df

def build_eps_quarterly_df(stock):
    earning_dict = fmpsdk.earnings_surprises(apikey, stock)
    eps_a = []
    eps_e = []
    dates = []
    this_df = pd.DataFrame(columns = ['Date', 'Actual_Earnings', 'Estimated_Earnings'])
    for json in earning_dict:
        dates.append(json['date'])
        eps_a.append(json['actualEarningResult'])
        eps_e.append(json['estimatedEarning'])

    this_df.Date = dates
    this_df.Actual_Earnings = eps_a
    this_df.Estimated_Earnings = eps_e
    this_df.Date = pd.to_datetime(this_df.Date)  
    return this_df

def build_stock_graph_df(stock, apikey):
    this_df = pd.DataFrame(columns=['Date', 'Close', 'Symbol'])
    price_history = fmpsdk.historical_price_full(apikey, stock, from_date=start,
                             to_date=end
                    )
    dates = []
    closes = []
    symbol = []
    for dictionary in price_history['historical']:
        dates.append(dictionary['date'])
        closes.append(dictionary['close'])
        symbol.append(stock)
    this_df.Symbol = symbol
    this_df.Date = dates
    this_df.Close = closes
    return this_df

def build_time_series_df(stock):   
    today = datetime.now()
    end_date = today - timedelta(days=200)
    day_stock = fmpsdk.historical_price_full(apikey, stock, from_date = end_date,
                                            to_date = today)
    
    day_df = pd.DataFrame(columns = ['Date', 'Close', 'High', 'Change Over Time',
                                    'Volume'])
    dates = []
    closes = []
    highs = []
    change_over_time = []
    changes = []
    volumes = []
    


    for day in day_stock['historical']:
        dates.append(day['date'])
        closes.append(day['close'])
        highs.append(day['high'])
        change_over_time.append(day['changeOverTime'])
        volumes.append(day['volume'])
        

    day_df.Date = dates
    day_df.Close = closes
    day_df.High = highs
    day_df['Change Over Time'] = change_over_time
    day_df.Volume = volumes

    day_df.Date = day_df.Date.apply(lambda dat: pd.to_datetime(dat))
    #day_df.Date = day_df.Date.apply(lambda dat: str(dat)[5:10])
 
    return day_df

def build_xgboost_df(symbol):
    
    prediction_df = pd.DataFrame(columns = ['Close_1day',
                                            'Close_2day',
                                            'Close_3day',
                                            #'Close_4day',
                                            #'Close_5day',
                                            'High_1day',
                                            'High_2day',
                                            'High_3day',
                                            #'High_4day',
                                           #'High_5day',
                                            'Cot_1day',
                                            'Cot_2day',
                                           'Cot_3day',
                                           # 'Cot_4day',
                                            #'Cot_5day',
                                           'Vol_1day',
                                            'Vol_2day',
                                            'Vol_3day',
                                            #'Vol_4day',
                                            #'Vol_5day',
                                            'value',
                                            'Date'
                                       ])
    stock_df = build_time_series_df(symbol)
    prediction_df.value = stock_df.Close[0:(stock_df.shape[0])-5]
    
    prediction_df.Date = stock_df.Date
    
    day_1_close = []
    day_2_close = []
    day_3_close = []
    day_4_close = []
    day_5_close = []

    High_1day = []
    High_2day = []
    High_3day = []
    High_4day = []
    High_5day = []

    Cot_1day = []
    Cot_2day = []
    Cot_3day = []
    Cot_4day= []
    Cot_5day= []

    Vol_1day = []
    Vol_2day = []
    Vol_3day = []
    Vol_4day = []
    Vol_5day = []
    for i in range(len(prediction_df.value)):

        day_1_close.append(stock_df.Close[i+1])
        day_2_close.append(stock_df.Close[i+2])
        day_3_close.append(stock_df.Close[i+3])
#         day_4_close.append(stock_df.Close[i+4])
#         day_5_close.append(stock_df.Close[i+5])

        High_1day.append(stock_df.High[i+1])
        High_2day.append(stock_df.High[i+2])
        High_3day.append(stock_df.High[i+3])
#         High_4day.append(stock_df.High[i+4])
#         High_5day.append(stock_df.High[i+5])

        Cot_1day.append(stock_df['Change Over Time'][i+1])
        Cot_2day.append(stock_df['Change Over Time'][i+2])
        Cot_3day.append(stock_df['Change Over Time'][i+3])
#         Cot_4day.append(stock_df['Change Over Time'][i+4])
#         Cot_5day.append(stock_df['Change Over Time'][i+5])

        Vol_1day.append(stock_df.Volume[i+1])
        Vol_2day.append(stock_df.Volume[i+2])
        Vol_3day.append(stock_df.Volume[i+3])
#         Vol_4day.append(stock_df.Volume[i+4])
#         Vol_5day.append(stock_df.Volume[i+5])

    prediction_df.Close_1day = day_1_close
    prediction_df.Close_2day = day_2_close
    prediction_df.Close_3day = day_3_close
#     prediction_df.Close_4day = day_4_close
#     prediction_df.Close_5day = day_5_close

    prediction_df.High_1day = High_1day
    prediction_df.High_2day= High_2day
    prediction_df.High_3day= High_3day
#     prediction_df.High_4day= High_4day
#     prediction_df.High_5day= High_5day

    prediction_df.Cot_1day = Cot_1day
    prediction_df.Cot_2day = Cot_2day
    prediction_df.Cot_3day = Cot_3day
#     prediction_df.Cot_4day = Cot_4day
#     prediction_df.Cot_5day = Cot_5day

    prediction_df.Vol_1day = Vol_1day
    prediction_df.Vol_2day = Vol_2day
    prediction_df.Vol_3day = Vol_3day
#     prediction_df.Vol_4day = Vol_4day
#     prediction_df.Vol_5day = Vol_5day

    prediction_df['News_Sentiment_1day'] = None
    news_df = vader_sentiment_analysis_prediction(apikey, symbol)
    prediction_df.News_Sentiment_1day = prediction_df.Date.apply(
        lambda dat: np.mean(trim_news_dates(str(dat), news_df).CompoundAvg))

    
    prediction_df.News_Sentiment_1day.fillna(0, inplace=True)
    return prediction_df

def build_general_model_daily_prediction_df(stock):
    prediction_df = build_xgboost_df(stock)

    test_data, train_data = prediction_df[0:int(len(prediction_df)*.1)], \
                            prediction_df[int(len(prediction_df)*.1):]
    
    X_test = test_data.drop(columns = ['Date', 'value'])

    y_train = train_data.value
    X_train = train_data.drop(columns = ['Date', 'value'])
    y_test = test_data.value

    xg = GradientBoostingRegressor()
    xg.fit(X_train, y_train)
    y_pred = pd.Series(xg.predict(X_test))
    mae = np.sqrt(mean_squared_error(y_test, y_pred))
    
    xgboost_prediction_df = pd.DataFrame(columns = ['value', 'data_category', 'Date'])
    combined_list = pd.concat([y_test, y_train, y_pred])
    combined_list.reset_index(drop = True, inplace = True)

    xgboost_prediction_df.value = combined_list
    xgboost_prediction_df.Date = prediction_df.Date



    test_cutoff = len(y_test)
    pred_cutoff = len(y_test) + len(y_train)
    counter = 0
    for i in range(len(combined_list)):
        if (counter > test_cutoff)&(counter < pred_cutoff):
            xgboost_prediction_df.data_category[i] = 'train'
        elif (counter <= test_cutoff):
            xgboost_prediction_df.data_category[i] = 'test'
        elif (counter >= pred_cutoff):
            xgboost_prediction_df.data_category[i] = 'pred'
        counter += 1
            
    pred_dates = xgboost_prediction_df.iloc[:len(y_test)].Date
    count = 0
    i = 0
    for date in xgboost_prediction_df.Date:
        if str(date)[0] == 'N':
            xgboost_prediction_df.Date[count] = pred_dates[i]
            i += 1
        count += 1
        
    
    return xgboost_prediction_df, xg, prediction_df, mae

def make_tomorrow_pred(stock, series, model):
    cols = ['close_1day', 'close_2day','close_3day', 'high_1day',
              'high_2day', 'high_3day', 'cot_1day', 'cot_2day',
              'cot_3day', 'vol_1day', 'vol_2day', 'vol_3day', 'news_sent1day']
    X = pd.DataFrame(columns = cols)
    yesterday_data = fmpsdk.historical_price_full(apikey, stock)['historical'][0]
    
    vsa = vader_sentiment_analysis_prediction(apikey, stock, limit = 20)
    
    X.news_sent1day = pd.Series(np.mean(trim_news_dates(datetime.today().strftime('%Y-%m-%d'), vsa).CompoundAvg))
    X.news_sent1day.fillna(0, inplace=True)
    X.close_1day = pd.Series(yesterday_data['close'])
    X.close_2day = pd.Series(series.Close_1day[0])
    X.close_3day = pd.Series(series.Close_2day[0])

    X.high_1day = pd.Series(yesterday_data['high'])
    X.high_2day = pd.Series(series.High_1day[0])
    X.high_3day = pd.Series(series.High_2day[0])

    X.cot_1day = pd.Series(yesterday_data['changeOverTime'])
    X.cot_2day = pd.Series(series.Cot_1day[0])
    X.cot_3day = pd.Series(series.Cot_2day[0])

    X.vol_1day = pd.Series(yesterday_data['volume'])
    X.vol_2day = pd.Series(series.Vol_1day[0])
    X.vol_3day = pd.Series(series.Vol_2day[0])

    pred = model.predict(X)
    
    return pred

def build_time_series_hourly_df(symbol):

    hourly_stock = fmpsdk.historical_chart(apikey, symbol, "1hour")
    
    hourly_df = pd.DataFrame(columns = ['Time', 'Close', 'High', 'Volume', 'Low'])
    opens = []
    times = []
    closes = []
    highs = []
    volumes = []
    lows = []
    for hour in hourly_stock:
        times.append(hour['date'])
        opens.append(hour['open'])
        closes.append(hour['close'])
        highs.append(hour['high'])
        volumes.append(hour['volume'])
        lows.append(hour['low'])

    hourly_df.Time = times
    hourly_df.Close = closes
    hourly_df.High = highs
    hourly_df.Volume = volumes
    hourly_df.Low = lows

 
    return hourly_df

def build_xgboost_hourly_df(symbol):
    
    prediction_df = pd.DataFrame(columns = ['Close_1hour',
                                            'Close_2hour',
                                            'Close_3hour',
                                            #'Close_4day',
                                            #'Close_5day',
                                            'High_1hour',
                                            'High_2hour',
                                            'High_3hour',
                                            #'High_4day',
                                           #'High_5day',
                                            'Low_1hour',
                                            'Low_2hour',
                                           'Low_3hour',
                                           # 'Cot_4day',
                                            #'Cot_5day',
                                           'Vol_1hour',
                                            'Vol_2hour',
                                            'Vol_3hour',
                                            #'Vol_4day',
                                            #'Vol_5day',
                                            'value',
                                            'Time'
                                       ])
    stock_df = build_time_series_hourly_df(symbol)
    prediction_df.value = stock_df.Close[0:(stock_df.shape[0])-5]
    prediction_df.Time = stock_df.Time
    day_1_close = []
    day_2_close = []
    day_3_close = []
    day_4_close = []
    day_5_close = []

    High_1day = []
    High_2day = []
    High_3day = []
    High_4day = []
    High_5day = []

    Low_1day = []
    Low_2day = []
    Low_3day = []
    Low_4day= []
    Low_5day= []

    Vol_1day = []
    Vol_2day = []
    Vol_3day = []
    Vol_4day = []
    Vol_5day = []
    for i in range(len(prediction_df.value)):

        day_1_close.append(stock_df.Close[i+1])
        day_2_close.append(stock_df.Close[i+2])
        day_3_close.append(stock_df.Close[i+3])
#         day_4_close.append(stock_df.Close[i+4])
#         day_5_close.append(stock_df.Close[i+5])

        High_1day.append(stock_df.High[i+1])
        High_2day.append(stock_df.High[i+2])
        High_3day.append(stock_df.High[i+3])
#         High_4day.append(stock_df.High[i+4])
#         High_5day.append(stock_df.High[i+5])

        Low_1day.append(stock_df['Low'][i+1])
        Low_2day.append(stock_df['Low'][i+2])
        Low_3day.append(stock_df['Low'][i+3])
#         Cot_4day.append(stock_df['Change Over Time'][i+4])
#         Cot_5day.append(stock_df['Change Over Time'][i+5])

        Vol_1day.append(stock_df.Volume[i+1])
        Vol_2day.append(stock_df.Volume[i+2])
        Vol_3day.append(stock_df.Volume[i+3])
#         Vol_4day.append(stock_df.Volume[i+4])
#         Vol_5day.append(stock_df.Volume[i+5])

    prediction_df.Close_1hour = day_1_close
    prediction_df.Close_2hour = day_2_close
    prediction_df.Close_3hour = day_3_close
#     prediction_df.Close_4day = day_4_close
#     prediction_df.Close_5day = day_5_close

    prediction_df.High_1hour = High_1day
    prediction_df.High_2hour = High_2day
    prediction_df.High_3hour = High_3day
#     prediction_df.High_4day= High_4day
#     prediction_df.High_5day= High_5day

    prediction_df.Low_1hour = Low_1day
    prediction_df.Low_2hour = Low_2day
    prediction_df.Low_3hour = Low_3day
#     prediction_df.Cot_4day = Cot_4day
#     prediction_df.Cot_5day = Cot_5day

    prediction_df.Vol_1hour = Vol_1day
    prediction_df.Vol_2hour = Vol_2day
    prediction_df.Vol_3hour = Vol_3day
#     prediction_df.Vol_4day = Vol_4day
#     prediction_df.Vol_5day = Vol_5day

        
    return prediction_df

def build_general_model_hourly_prediction_df(stock):
    prediction_df = build_xgboost_hourly_df(stock)

    test_data, train_data = prediction_df[0:int(len(prediction_df)*.025)], \
                            prediction_df[int(len(prediction_df)*.025):]
    
    X_test = test_data.drop(columns = ['Time', 'value'])

    y_train = train_data.value
    X_train = train_data.drop(columns = ['Time', 'value'])
    y_test = test_data.value

    xg = GradientBoostingRegressor()
    xg.fit(X_train, y_train)
    y_pred = pd.Series(xg.predict(X_test))
    mae = np.sqrt(mean_squared_error(y_test, y_pred))
    
    xgboost_prediction_df = pd.DataFrame(columns = ['value', 'data_category', 'Time'])
    combined_list = pd.concat([y_test, y_train, y_pred])
    combined_list.reset_index(drop = True, inplace = True)

    xgboost_prediction_df.value = combined_list
    xgboost_prediction_df.Time = prediction_df.Time



    test_cutoff = len(y_test)
    pred_cutoff = len(y_test) + len(y_train)
    counter = 0
    for i in range(len(combined_list)):
        if (counter > test_cutoff)&(counter < pred_cutoff):
            xgboost_prediction_df.data_category[i] = 'train'
        elif (counter <= test_cutoff):
            xgboost_prediction_df.data_category[i] = 'test'
        elif (counter >= pred_cutoff):
            xgboost_prediction_df.data_category[i] = 'pred'
        counter += 1
            
    pred_dates = list(xgboost_prediction_df.iloc[:len(y_test)].Time)
    count = 0
    i = 0
    for date in xgboost_prediction_df.Time:
        if date is np.nan:
            xgboost_prediction_df.Time[count] = pred_dates[i]
            i += 1
        count += 1
        
    #return pred_dates, xgboost_prediction_df
    return xgboost_prediction_df, xg, prediction_df, mae

def make_next_hour_pred(stock, series, model):
    cols = ['close_1day', 'close_2day','close_3day', 'high_1day',
              'high_2day', 'high_3day', 'low_1day', 'low_2day',
              'low_3day', 'vol_1day', 'vol_2day', 'vol_3day']
    X = pd.DataFrame(columns = cols)

    last_hour_data = fmpsdk.historical_chart(apikey, 'AAPL', "1hour")[0]

    X.close_1day = pd.Series(last_hour_data['close'])
    X.close_2day = pd.Series(series.Close_1hour[0])
    X.close_3day = pd.Series(series.Close_2hour[0])

    X.high_1day = pd.Series(last_hour_data['high'])
    X.high_2day = pd.Series(series.High_1hour[0])
    X.high_3day = pd.Series(series.High_2hour[0])

    X.low_1day = pd.Series(last_hour_data['low'])
    X.low_2day = pd.Series(series.Low_1hour[0])
    X.low_3day = pd.Series(series.Low_2hour[0])

    X.vol_1day = pd.Series(last_hour_data['volume'])
    X.vol_2day = pd.Series(series.Vol_1hour[0])
    X.vol_3day = pd.Series(series.Vol_2hour[0])

    pred = model.predict(X)
    
    return pred

######################################################################################
external_stylesheets = [dbc.themes.BOOTSTRAP]
# Create an app.
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server
#app = DashProxy(prevent_initial_callbacks=True, transforms=[MultiplexerTransform()],
 #                   external_stylesheets=external_stylesheets)
# Create an app.
app.title = 'My App'

#############################################
#------------BACK END------------------------
end = datetime.now()
start = end - timedelta(days=100)
stock_sym = 'AAPL'

#Here I am making a list of all possible stock tickers
#I will use the list available_stocks to confirm that a user input is valid
available_stocks_dict = fmpsdk.available_traded_list(apikey)
available_stocks = []
for ticker in available_stocks_dict:
    available_stocks.append(ticker['symbol'])



#Header
row_one = html.Div(dbc.Row
    ([
        dbc.Col(html.H1(children = 'Stock Analysis Dashboard', style={'text-align': 'center'}),
        align='center', width={'offset':2}),
        dbc.Col(html.Div(

           [ daq.ToggleSwitch(id="side_by_side", 
             label="", 
             value = True,
             size = 30),
            html.Div(id='my-toggle-switch-output')]
    
            

        ),width = 2, align='right')
         ],class_name='g-0')        
)
row_two = html.Div(
    dbc.Row(
        [
        #Market Trends
        dbc.Col(
            html.Div([
                dcc.Dropdown(
                    id='graph_to_show',
                    value = 'SPOT',
                    options= [
                        {'label' : 'Stock Price Over Time', 'value': 'SPOT'},
                        {'label' : 'Annual Revenue', 'value': 'AR'},
                        {'label' : 'Earnings Per Share Annual', 'value' : 'EPS'},
                        {'label' : 'Genral Model Daily Prediction', 'value' : "GMD"},
                        {'label' : 'General Model Hourly Prediction', 'value' : "GMH"},
                        {'label' : 'Earnings Per Share Quarterly', 'value' : 'EPS_Q'}
                        
                    ],
                    searchable=False,
                    clearable=False
                   
                    
                    
                )


            ]), align="left", width=8, id='og_graph_to_show_col'

        ),
        #Enter Stock Symbol Will Go Here
        dbc.Col(
            html.Div([
                dcc.Input(id='input', value=stock_sym, type='text')

            ]), align="left", id='og_input_button_col'

            ),
            

        
        dbc.Col(
            html.Div([
                html.Button(n_clicks=0, children='Submit', id='submit-button'
                
            )]

        ), id='og_submit_button_col'),

        dbc.Col(
            html.Div([
                dcc.Dropdown(
                    id='graph_to_show_sbs',
                    value = 'SPOT',
                    options= [
                        {'label' : 'Stock Price Over Time', 'value': 'SPOT'},
                        {'label' : 'Annual Revenue', 'value': 'AR'},
                        {'label' : 'Earnings Per Share', 'value' : 'EPS'},
                        {'label' : 'Genral Model Daily Prediction', 'value' : "GMD"},
                        {'label' : 'General Model Hourly Prediction', 'value' : "GMH"},
                         {'label' : 'Earnings Per Share Quarterly', 'value' : 'EPS_Q'}
                        
                    ],
                    style = dict(display='none'),
                    searchable=False,
                    clearable=False

                    
                    
                )


            ]),
            width = 4


            #style = {'align':'right', 'offset':'2'}

        ),
        dbc.Col(
            html.Div([
                dcc.Input(id='input_sbs', value=stock_sym, type='text',
                 style = {'display':'none'})
                

            ]),
            width = 1
             

        ), 
        dbc.Col(
            html.Button(id='submit-button_sbs', n_clicks=0, children='Submit',
             style = dict(display='none'))

         ,width = 1),

        ], class_name = 'g-1'
        

    )
       
)






    



# Setup a simple html `div`.



df = build_stock_graph_df(stock_sym, apikey) 
og_fig = go.Figure([go.Scatter(x = df['Date'], y = df.Close,\
                     line = dict(color = 'firebrick', width = 4), name = df.Symbol[0])
                     ])
og_fig.update_layout(title = 'Prices over time for '+df.Symbol[0]+"    |    Current Price: "+str(get_current_price(df.Symbol[0])),
                      xaxis_title = 'Dates',
                      yaxis_title = 'Prices'
                      )

og_table = build_ratings_table(stock_sym)

row_three = html.Div(
    dbc.Row([
        dbc.Col(
            dcc.Graph(id = 'line_plot', figure=og_fig),
            width = 12, id = 'line_plot_col'
        ),
         dbc.Col(
            dcc.Graph(id = 'line_plot_sbs', figure=og_fig,  style = dict(display='none')),
            width = 6
           
        )]
    )
)

row_four = html.Div([
    dbc.Row([
        dbc.Col(
            html.Div([
                dcc.Dropdown(
                    id='table_to_show',
                    value = 'RT',
                    options= [
                        {'label' : 'Ratings', 'value': 'RT'},
                        {'label' : 'Recent News', 'value': 'RN'},
                        {'label' : 'Recent News Sentiment', 'value' : 'RNS'},
                        {'label' : 'Insider Trades', 'value' : 'IT'}
                    ],
                    searchable=False,
                    clearable=False
                    
                    
                )


            ]), id = 'og_table_to_show_col', width=10


        ),
        dbc.Col(
            html.Div([
                dcc.Input(id='quantity', value=10, type='number', 
                style = dict(display='none'
                            ))
                

            ]),
            
        ),
        dbc.Col(
            html.Div([
                dcc.Dropdown(
                    id='table_to_show_sbs',
                    value = 'RT',
                    options= [
                        {'label' : 'Ratings', 'value': 'RT'},
                        {'label' : 'Recent News', 'value': 'RN'},
                        {'label' : 'Recent News Sentiment', 'value' : 'RNS'},
                        {'label' : 'Insider Trades', 'value' : 'IT'}
                    ],
                    style = dict(display='none'),
                    searchable=False,
                    clearable=False
                    
                    
                )


            ])


        ,width=4),
        dbc.Col(
            html.Div([
                dcc.Input(id='quantity_sbs', value=10, type='number', style = dict(display='none'))
                

            ]),
            
        )
    ],class_name='g-0')
]
)








row_five = html.Div([
    dbc.Row([
        dbc.Col(
            html.Div([
                dbc.Table(
                #id='table',
                children=og_table,
                
                
                

                )


            ]), width = 12,  id ='table'

        ),
        dbc.Col(
             html.Div([
                dbc.Table(
                id = 'table_sbs',
                children = og_table,
                style = dict(display='none'),
               

                )

            ]), 

        )




    ], class_name = 'g-9')
])
alert = html.Div([

    dbc.Alert(
        "Please Choose A Valid Stock Symbol",
        id='Alert',
        dismissable=True,
        is_open=False,
        color='danger'
    ),
    dbc.Alert(
        "Please Choose A Valid Stock Symbol",
        id='Alert_sbs',
        dismissable=True,
        is_open=False,
        color='danger'

    )
])

######################################################################
####################### APP Layout ###################################
######################################################################

app.layout = html.Div(
    id='main_div',
    children=[alert, row_one, row_two, row_three, row_four, row_five],
    style={'width': '100%', 'margin': 'auto', 'padding': '0'}
)
######################################################################
####################### CAllBACKS  ###################################
######################################################################

#SIDE BY SIDE MODE BUTTON CALLBACK
@app.callback(
    #  Input: side_by_side button 
    #  OUTPUT: 
    # 

    Output('line_plot_sbs', 'style'),
    Output('input_sbs', 'style'),
    Output('graph_to_show_sbs', 'style'),
    Output('submit-button_sbs', 'style'),
    Output('table_to_show_sbs', 'style'),
    Output('table_sbs', 'style'),
    Output('side_by_side', 'children'),

    Output('line_plot_col', 'style'),
    Output('og_input_button_col', 'style'),
    Output('og_graph_to_show_col', 'style'),
    Output('og_submit_button_col', 'style'),
    Output('og_table_to_show_col', 'style'),
    Output('table', 'style'),
    Output('my-toggle-switch-output', 'children'),
    Output('side_by_side', 'label'),
    Output('quantity_sbs', 'style'),
    
    Input('side_by_side', 'value'),
    State('side_by_side', 'value'),
    Input('table_to_show_sbs', 'value')
    )
def side_by_side_mode(value, sbs ,table): 
    activate = dict(display='none')
    if sbs == False:
        if table == 'RN' :
            activate = dict(width = '50%')
        elif table == 'RNS':
            activate = dict(width = '50%')
        elif table == 'IT':
            activate = dict(width = '50%')
    else:
        activate = dict(display='none')

    #Check if currently is side by side by mode
    #at time of button click
    if value == True:
        #Turns side by side mode off
        to_return = dict(width='0%', display = 'none')
        #Update global boolean sbs 
        update_sbs(False)
        return to_return, to_return, to_return, to_return, to_return, to_return, 'Side By Side Mode: OFF', \
            dict(width = '100%'), dict(width = '20%'), dict(width = '75%'), dict(width = '5%'), dict(width = '80%'),\
            dict(width = '100%'), False, 'SIDE BY SIDE: OFF', activate

    elif value == False:
        to_return = dict()
        update_sbs(True)
        return to_return, to_return, to_return, to_return, to_return, to_return, 'Side By Side Mode: ON', \
            dict(width = '50%'), dict(width = '10%'), dict(width = '32%'),\
            dict(width = '2%'), dict(width = '35%'),\
            dict(width = '50%'), True, 'SIDE BY SIDE: ON', activate
    





@app.callback(
    Output('line_plot', 'figure'),
    Output('input', 'value'),
    Output('table', 'children'),
    Output('quantity', 'style'),
    Output('Alert', 'is_open'),
    State('Alert', 'is_open'),
    Input('quantity', 'value'),
    Input('submit-button', 'n_clicks'),
    State('input', 'value'),
    Input('graph_to_show', 'value'),
    Input('table_to_show', 'value')
    )
def update_graph(is_open, quantity, n_clicks, value, chart, table):
    #Initialize acivate to empty this object is used to change the quantity value
    #and will only be activated under certain conditions.
    activate = dict(display='none')
    if value not in available_stocks:
        p_df = build_stock_graph_df(stock_sym, apikey)
        fig = go.Figure([go.Scatter(x = p_df['Date'], y = p_df.Close,\
                        line = dict(color = 'firebrick', width = 4), name = p_df.Symbol[0])
                        ])
        fig.update_layout(title = 'Prices over time for '+p_df.Symbol[0]+"    |    Current Price: "+str(get_current_price(p_df.Symbol[0])),
                        xaxis_title = 'Dates',
                        yaxis_title = 'Prices'
                        )
        the_table = build_ratings_table(stock_sym)
            
        return fig, value, the_table, activate,True
    #Valid stock symbol inputted, update the global variable
    update_stock_sym(value)
    #Table Choices
    if table == 'RT':
        the_table = build_ratings_table(value)  
    elif table == 'RN':
        the_table = build_news_table(value, quantity)
        activate = dict(width = '50%')
    elif table == 'RNS':
        the_table = vader_sentiment_analysis(value, quantity)
        activate = dict(width = '50%')
    elif table == 'IT':
        the_table = build_insider_table(value, quantity)
        activate = dict(width = '50%')
    if chart == 'AR':
        chart_df = get_annual_revenue_df(stock_sym)
        #Some companies don't have revenue data because they might be too young
        if type(chart_df) == int:
            fig = px.bar(title= 'There is no Data For '+stock_sym+' yet.....')
        else:    

            fig = px.bar(chart_df, x='Dates', y=['Revenue', 'NetIncome'], 
                    title= 'Annual Financials For: '+chart_df.Symbol[0], barmode='group',
                    
                    )
        return fig, value, the_table, activate, False
    elif chart == 'SPOT':
        p_df = build_stock_graph_df(stock_sym, apikey)
        fig = go.Figure([go.Scatter(x = p_df['Date'], y = p_df.Close,\
                        line = dict(color = 'firebrick', width = 4), name = p_df.Symbol[0])
                        ])
        fig.update_layout(title = 'Prices over time for '+p_df.Symbol[0]+"    |    Current Price: "+str(get_current_price(p_df.Symbol[0])),
                        xaxis_title = 'Dates',
                        yaxis_title = 'Prices'
                        )
            
        return fig, value, the_table, activate, False
    elif chart == 'EPS':
        p_df = build_eps_df(stock_sym)
        fig = px.scatter(p_df, x='Date', y=['EarningPerShare'], 
             title= 'Annual Earnings Per Share & Trendline For: ' +stock_sym, trendline='ols')
        return fig, value, the_table, activate, False
    elif chart == 'EPS_Q':
        p_df = build_eps_quarterly_df(stock_sym)
        fig = px.scatter(p_df, x='Date', y=['Actual_Earnings', 'Estimated_Earnings'], 
             title= 'Quarterly Actual/Estimated Earnings Per Share For: '+stock_sym)
        fig.update_traces(marker=dict(size=12,
                              line=dict(width=2,
                                        color='DarkSlateGrey')),
                  selector=dict(mode='markers'))
        return fig, value, the_table, activate, False
    elif chart == 'GMD':
        xg_df, xg, pred_df, mae = build_general_model_daily_prediction_df(stock_sym)
        fig = px.line(xg_df, x='Date', y='value', color='data_category', markers=True,
        title = stock_sym+' | Next Day Close Prediction: '
            + str(make_tomorrow_pred(stock_sym, pred_df, xg))+' +/- '+str(mae))
           
        return fig, value, the_table, activate, False
    elif chart == 'GMH':
        xg_df, xg, pred_df, mae = build_general_model_hourly_prediction_df(stock_sym)
        fig = px.line(xg_df, x='Time', y='value', color='data_category', markers=True,
        title = stock_sym+' | Next Hour Close Prediction: '
            + str(make_next_hour_pred(stock_sym, pred_df, xg))+' +/- '+str(mae))
        return fig, value, the_table, activate, False




@app.callback(
    Output('line_plot_sbs', 'figure'),
    Output('input_sbs', 'value'),
    Output('table_sbs', 'children'),
    #Output('quantity_sbs', 'style'),
    Output('Alert_sbs', 'is_open'),

    State('Alert_sbs', 'is_open'),
    Input('quantity_sbs', 'value'),
    Input('submit-button_sbs', 'n_clicks'),
    State('input_sbs', 'value'),
    Input('graph_to_show_sbs', 'value'),
    Input('table_to_show_sbs', 'value'),
    
    )
def update_graph_sbs(is_open, quantity, n_clicks, value, chart, table):
    #Initialize acivate to empty this object is used to change the quantity value
    #and will only be activated under certain conditions.
    activate = dict(display='none')
    if value not in available_stocks:
        p_df = build_stock_graph_df(stock_sym, apikey)
        fig = go.Figure([go.Scatter(x = p_df['Date'], y = p_df.Close,\
                        line = dict(color = 'firebrick', width = 4), name = p_df.Symbol[0])
                        ])
        fig.update_layout(title = 'Prices over time for '+p_df.Symbol[0]+"    |    Current Price: "+str(get_current_price(p_df.Symbol[0])),
                        xaxis_title = 'Dates',
                        yaxis_title = 'Prices'
                        )
        the_table = build_ratings_table(stock_sym)
            
        return fig, value, the_table,True
    #Valid stock symbol inputted, update the global variable
    update_stock_sym(value)
    #Table Choices
    if table == 'RT':
        the_table = build_ratings_table(value)  
    elif table == 'RN':
        the_table = build_news_table(value, quantity)
        
    elif table == 'RNS':
        the_table = vader_sentiment_analysis(value, quantity)
        
    elif table == 'IT':
        the_table = build_insider_table(value, quantity)
        
    if chart == 'AR':
        chart_df = get_annual_revenue_df(stock_sym)
        #Some companies don't have revenue data because they might be too young
        if type(chart_df) == int:
            fig = px.bar(title= 'There is no Data For '+stock_sym+' yet.....')
        else:    

            fig = px.bar(chart_df, x='Dates', y=['Revenue', 'NetIncome'], 
                    title= 'Annual Financials For: '+chart_df.Symbol[0], barmode='group',
                    
                    )
        return fig, value, the_table, False
    elif chart == 'SPOT':
        p_df = build_stock_graph_df(stock_sym, apikey)
        fig = go.Figure([go.Scatter(x = p_df['Date'], y = p_df.Close,\
                        line = dict(color = 'firebrick', width = 4), name = p_df.Symbol[0])
                        ])
        fig.update_layout(title = 'Prices over time for '+p_df.Symbol[0]+"    |    Current Price: "+str(get_current_price(p_df.Symbol[0])),
                        xaxis_title = 'Dates',
                        yaxis_title = 'Prices'
                        )
            
        return fig, value, the_table, False
    elif chart == 'EPS':
        p_df = build_eps_df(stock_sym)
        fig = px.scatter(p_df, x='Date', y=['EarningPerShare'], 
             title= 'Annual Earnings Per Share & Trendline For: ' +stock_sym, trendline='ols')
        return fig, value, the_table, False
    elif chart == 'EPS_Q':
        p_df = build_eps_quarterly_df(stock_sym)
        fig = px.scatter(p_df, x='Date', y=['Actual_Earnings', 'Estimated_Earnings'], 
             title= 'Quarterly Actual/Estimated Earnings Per Share For: '+stock_sym)
        fig.update_traces(marker=dict(size=12,
                              line=dict(width=2,
                                        color='DarkSlateGrey')),
                  selector=dict(mode='markers'))
        return fig, value, the_table, False
        
    
    elif chart == 'GMD':
        xg_df, xg, pred_df, mae = build_general_model_daily_prediction_df(stock_sym)
        fig = px.line(xg_df, x='Date', y='value', color='data_category', markers=True,
        title = stock_sym+' | Next Day Close Prediction: '
            + str(make_tomorrow_pred(stock_sym, pred_df, xg))+' +/- '+str(mae))
           
        return fig, value, the_table, False

    elif chart == 'GMH':
        xg_df, xg, pred_df, mae = build_general_model_hourly_prediction_df(stock_sym)
        fig = px.line(xg_df, x='Time', y='value', color='data_category', markers=True,
        title = stock_sym+' | Next Hour Close Prediction: '
            + str(make_next_hour_pred(stock_sym, pred_df, xg))+' +/- '+str(mae))
        return fig, value, the_table, False



# Run the app.
if __name__ == '__main__':
    app.run_server(debug=True)
    #debug=True, port=8000, host='127.0.0.1'

  