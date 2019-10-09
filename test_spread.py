import gspread
from oauth2client.service_account import ServiceAccountCredentials

scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']

credentials = ServiceAccountCredentials.from_json_keyfile_name('./get-text-from-google-home-5fab0d67bbdb.json', scope)
gc = gspread.authorize(credentials)
wks = gc.open('Google Assistant Commands').sheet1

print(wks.acell('B1'))