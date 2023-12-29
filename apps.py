import streamlit as st
import pandas as pd
import streamlit as st
from mlxtend.frequent_patterns import apriori, association_rules
from PIL import Image
import plotly.express as px


# !pip install mlxtend
# pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)

st.set_page_config(page_title="MorMen Müşteri Analitiği",
                   page_icon="🤵‍♂️",
                   layout="wide")

st.image("Miuul - Project/logo1.jpg", width=100)
tab_vis, tab_segment, tab_recommen = st.tabs(["Dashboard", "Müşteri Segment Tahmini", "Ürün Öneri Sistemi"])


###################################
######## DASHBORD SEKMESİ #########

@st.cache_data
def import_data2(file_path):
    data2 = pd.read_excel(file_path)
    return data2

def remove_columns_and_values(dataframe):
    drop_cols = ['InvoiceTime', 'InvoiceDate', 'OtherPayments']
    dataframe = dataframe.drop(drop_cols, axis=1)

    drop_value = ['ADANA OPTIMUM PANAYIR', 'PANAYIR OPTIMUM', 'URFA PANAYIR', 'BİLİR GİYİM SAN.TİC.A.Ş.']
    dataframe = dataframe[~dataframe['NameSurname'].isin(drop_value)]
    dataframe = dataframe[dataframe['OfficeName'] != 'KURUMSAL MAĞAZA']
    dataframe = dataframe[~dataframe['ProductDescription'].isin(['Bez Büyük Boy Çanta', 'Bez Orta Boy Çanta'])]
    dataframe = dataframe[~dataframe['ProductType'].isin(['ÇORAP', 'Classic Cep Parfümü', 'HEDİYE KARTI', 'MASKE'])]

    #if dataframe['CustomerID'].str.contains('-').any():
        #dataframe['CustomerID'] = dataframe['CustomerID'].str.replace('-', '')
    dataframe = dataframe.dropna()

    return dataframe


def create_new_variables(dataframe):
    colors = {1: 'Siyah', 2: 'Siyah', 50: 'Füme', 53: 'Antrasit', 100: 'Gri', 102: 'Grimel', 103: 'AÇIK GRİ',
              105: 'Orta Gri', 156: 'Somon', 200: 'Lacivert', 201: 'KOYU LACİVERT', 210: 'Marine', 218: 'Parlement',
              300: 'Mavi', 308: 'İndigo', 309: 'Mor', 311: 'A. MAVİ', 312: 'Saks Mavisi', 370: 'Mürdüm', 400: 'KAHVE',
              401: 'A-Kahverengi', 403: 'Kahverengi - Lacivert', 405: 'AÇIK KAHVE', 406: 'Fındık', 408: 'Camel',
              411: 'AÇIK KAHVE', 450: 'ORANJ', 453: 'Yavruağzı', 454: 'Ekru', 550: 'Kırmızı', 600: 'Beyaz',
              650: 'Krem rengi', 653: 'Bej', 700: 'Yeşil', 701: 'Açık Yeşil', 704: 'Su Yeşili', 710: 'Haki',
              750: 'Bordo', 752: 'Koyu Bordo', 870: 'Taş', 925: 'Muhtelif'}

    dataframe['ColorName'] = dataframe['ColorID'].map(colors)
    dataframe.insert(12, 'ColorName', dataframe.pop('ColorName'))

    dataframe['ColorID'] = dataframe['ColorID'].astype(int)

    dataframe['Hour'] = dataframe['DateTime'].dt.hour
    dataframe['DayName'] = dataframe['DateTime'].dt.day_name()

    dataframe['Month'] = dataframe['DateTime'].dt.month
    dataframe['MonthName'] = dataframe['DateTime'].dt.month_name()

    dataframe['PricePerUnit'] = dataframe['TotalPrice'] / dataframe['Amount']

    return dataframe


file_path = "Miuul - Project/Mağaza Satış 2023.xlsx"  # Veri seti dosya yolu
data2 = import_data2(file_path)
# Remove columns and values
data2 = remove_columns_and_values(data2)
# Create new variables
data2 = create_new_variables(data2)

# TAB VIS

col_left, col_center, col_right = tab_vis.columns(3)

## grafik 1

col_left.subheader("En Fazla Satın Alınan 10 Ürün")
top_products = data2.groupby('ProductType')['Amount'].sum().sort_values(ascending=False).head(10)
colors = px.colors.qualitative.Plotly[:len(top_products)]
fig1 = px.bar(
    top_products,
    x=top_products.index,
    y='Amount',
    color=top_products.index,
    color_discrete_sequence=colors
)

col_left.plotly_chart(fig1, use_container_width=True)


## grafik 2

col_center.subheader("Cirosu En Yüksek 10 Mağaza")
top_store = data2.groupby('OfficeName')['TotalPrice'].sum().sort_values(ascending=False).head(10)
colors = px.colors.qualitative.Plotly[:len(top_products)]
fig2 = px.bar(
    top_store.reset_index(),
    x='OfficeName',
    y='TotalPrice',
    color='OfficeName',
    color_discrete_sequence=colors
)

col_center.plotly_chart(fig2, use_container_width=True)

## grafik 3

col_right.subheader("En Fazla Ürün Satan 10 Mağaza")
top_product_store = data2.groupby('OfficeName')['Amount'].sum().sort_values(ascending=False).head(10)
colors = px.colors.qualitative.Plotly[:len(top_products)]
fig3 = px.bar(
    top_product_store.reset_index(),
    x='OfficeName',
    y='Amount',
    color='OfficeName',
    color_discrete_sequence=colors
)

col_right.plotly_chart(fig3, use_container_width=True)


# Grafik 4


# Verileri hazırla
# Verileri hazırla
filtered_df = data2[(data2['Hour'] >= 9) & (data2['Hour'] <= 21)]
filtered_df['DayName'] = data2['DateTime'].dt.day_name()
heatmap_data = filtered_df.pivot_table(index='DayName', columns='Hour', values='TotalPrice', aggfunc='sum')
days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
heatmap_data = heatmap_data.reindex(index=days_order, fill_value=0)
cat_type = pd.CategoricalDtype(categories=days_order, ordered=True)
filtered_df['DayName'] = filtered_df['DayName'].astype(cat_type)

tab_vis.subheader("Toplam Satış - Gün ve Saat (9:00 - 21:00)")

# Plotly Heatmap
fig4 = px.imshow(
    heatmap_data,
    labels=dict(x="Saat", y="Gün", color="Toplam Satış"),
    width=800,
    height=600,
    color_continuous_scale="viridis",
    color_continuous_midpoint=heatmap_data.values.max() / 2
)

# Kategori sıralamasını uygula
fig4.update_xaxes(categoryorder='array', categoryarray=heatmap_data.columns)
fig4.update_yaxes(categoryorder='array', categoryarray=heatmap_data.index)


# Görselleştirmeyi göster
tab_vis.plotly_chart(fig4, use_container_width=True)



# Grafik 5

data2['DateTime'] = pd.to_datetime(data2['DateTime'])

# Toplam kazancı zaman içinde grupla ve biriktir
total_revenue_over_time = data2.groupby(data2['DateTime'].dt.to_period("M"))['TotalPrice'].sum()
col_left.subheader("Aylara Göre Toplam Kazanç")

# Plotly Line Chart
fig5 = px.line(
    x=total_revenue_over_time.index.astype(str),
    y=total_revenue_over_time.values,
    labels=dict(x="Ay", y="Toplam Kazanç"),
    line_shape="linear",
    markers=True
)

# Görselleştirmeyi göster
col_left.plotly_chart(fig5, use_container_width=True)

# Grafik 6
# Renk bazında toplam satış miktarını hesapla ve sırala
top_color = data2.groupby('ColorName')['Amount'].sum().sort_values(ascending=False).head(3).reset_index()
# Farklı renkler için bir renk paleti oluştur
colors = px.colors.sequential.Viridis[:len(top_color)]

col_center.subheader("En Çok Tercih Edilen Renkler")

# Plotly Bar Chart
fig6 = px.bar(
    top_color,
    x='ColorName',
    y='Amount',
    color_continuous_scale=colors,
    labels=dict(ColorName="Renk", Amount="Toplam Satış")
)
# X-etiketlerini düzenleme
fig6.update_xaxes(tickangle=45, tickmode='array', tickvals=top_color.index, ticktext=top_color['ColorName'])
# Görselleştirmeyi göster
col_center.plotly_chart(fig6, use_container_width=True)


# Grafik 8

col_left.subheader("En Çok Kazandıran İlk 25 Müşteri")

# Müşteri bazında toplam kazancı hesapla ve sırala
top_earning_customers = data2.groupby('CustomerID')['TotalPrice'].sum().sort_values(ascending=False).head(25).reset_index()

# Plotly Bar Chart
fig8 = px.bar(
    top_earning_customers,
    x='CustomerID',
    y='TotalPrice',
    color='CustomerID',
    labels=dict(CustomerID="Müşteri ID", TotalPrice="Toplam Kazanç"),
)

# Görselleştirmeyi göster
col_left.plotly_chart(fig8, use_container_width=True)


# Grafik 9

# Müşteri bazında toplam kazancı hesapla ve sırala
top_earning_customers = data2.groupby('CustomerID')['Amount'].sum().sort_values(ascending=False).head(25).reset_index()
col_center.subheader("En Çok Ürün Satın Alan İlk 25 Müşteri")

# Plotly Bar Chart
fig9 = px.bar(
    top_earning_customers,
    x='CustomerID',
    y='Amount',
    color='CustomerID',
    labels=dict(CustomerID="Müşteri ID", TotalPrice="Toplam Kazanç")
)

# Görselleştirmeyi göster
col_center.plotly_chart(fig9, use_container_width=True)

####################################
# MÜŞTERİ SEGMENTASYONU SEKMESİ
####################################
import seaborn as sns
import pandas as pd
import streamlit as st
import datetime as dt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from yellowbrick.cluster import KElbowVisualizer
from sklearn.metrics import silhouette_score
from pandas.api.types import CategoricalDtype
from sklearn.preprocessing import StandardScaler
import warnings

warnings.simplefilter(action="ignore")

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.2f' % x)

@st.cache_data
def import_data3(file_path):
        # Veri Setinin Import Edilmesi
        data3 = pd.read_excel(file_path)
        return data3
def outlier_thresholds(dataframe, variable):
        # Aykırı değerleri tespit etmek için eşik değerleri hesaplayan fonksiyon
        quartile1 = dataframe[variable].quantile(0.01)
        quartile3 = dataframe[variable].quantile(0.99)
        interquantile_range = quartile3 - quartile1
        up_limit = quartile3 + 1.5 * interquantile_range
        low_limit = quartile1 - 1.5 * interquantile_range
        return low_limit, up_limit
def replace_with_thresholds(dataframe, variable):
        # Aykırı değerleri eşik değerleri ile değiştiren fonksiyon
        low_limit, up_limit = outlier_thresholds(dataframe, variable)
        dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
        dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit
def grab_col_names(dataframe, cat_th=10, car_th=50):
        """

        It gives the names of categorical, numerical and categorical but cardinal variables in the data set.
        Note: Categorical variables with numerical appearance are also included in the categorical variables.

        Parameters
        ------
            dataframe: dataframe
                    dataframe used
            cat_th: int, optional
                    threshold value for variables that are numeric but categorical
            car_th: int, optinal
                    threshold value for variables that are categorical but cardinal


        Returns
        ------
            cat_cols: list
                    Categorical variable list
            num_cols: list
                    Numeric variable list
            cat_but_car: list
                    List of cardinal variables with categorical view

        Examples
        ------
            import seaborn as sns
            df = sns.load_dataset("iris")
            print(grab_col_names(df))


        Notes
        ------
            cat_cols + num_cols + cat_but_car = total number of variables
            num_but_cat variables are in cat_cols.
            The sum of the 3 lists that return equals the total number of variables: cat_cols + num_cols + cat_but_car = otal number of variables

        """

        # cat_cols, cat_but_car
        cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
        num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                       dataframe[col].dtypes != "O"]
        cat_but_car = [col for col in dataframe.columns if
                       dataframe[col].dtypes == "O" and dataframe[col].nunique() > car_th]
        cat_cols = cat_cols + num_but_cat
        cat_cols = [col for col in cat_cols if col not in cat_but_car]

        # num_cols
        num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
        num_cols = [col for col in num_cols if col not in num_but_cat]

        print(f"Observations: {dataframe.shape[0]}")
        print(f"Variables: {dataframe.shape[1]}")
        print(f'cat_cols: {len(cat_cols)}')
        print(f'num_cols: {len(num_cols)}')
        print(f'cat_but_car: {len(cat_but_car)}')
        print(f'num_but_cat: {len(num_but_cat)}')
        return cat_cols, num_cols, cat_but_car
def preprocess_data(dataframe):
        # Veri ön işleme adımlarını içeren fonksiyon
        drop_cols = ['InvoiceTime', 'InvoiceDate', 'OtherPayments']
        dataframe = dataframe.drop(drop_cols, axis=1)
        # df'den çıkarılacak değerler
        drop_value = ['ADANA OPTIMUM PANAYIR', 'PANAYIR OPTIMUM', 'URFA PANAYIR', 'BİLİR GİYİM SAN.TİC.A.Ş.']
        dataframe = dataframe[~dataframe['NameSurname'].isin(drop_value)]
        dataframe = dataframe[dataframe['OfficeName'] != 'KURUMSAL MAĞAZA']
        dataframe = dataframe[~dataframe['ProductDescription'].isin(['Bez Büyük Boy Çanta', 'Bez Orta Boy Çanta'])]
        dataframe = dataframe[~dataframe['ProductType'].isin(['ÇORAP', 'Classic Cep Parfümü', 'HEDİYE KARTI', 'MASKE'])]
        dataframe.isnull().sum()
        dataframe = dataframe.dropna()
        colors = {1: 'Siyah',
                  2: 'Siyah',
                  50: 'Füme',
                  53: 'Antrasit',
                  100: 'Gri',
                  102: 'Grimel',
                  103: 'AÇIK GRİ',
                  105: 'Orta Gri',
                  156: 'Somon',
                  200: 'Lacivert',
                  201: 'KOYU LACİVERT',
                  210: 'Marine',
                  218: 'Parlement',
                  300: 'Mavi',
                  308: 'İndigo',
                  309: 'Mor',
                  311: 'A. MAVİ',
                  312: 'Saks Mavisi',
                  370: 'Mürdüm',
                  400: 'KAHVE',
                  401: 'A-Kahverengi',
                  403: 'Kahverengi - Lacivert',
                  405: 'AÇIK KAHVE',
                  406: 'Fındık',
                  408: 'Camel',
                  411: 'AÇIK KAHVE',
                  450: 'ORANJ',
                  453: 'Yavruağzı',
                  454: 'Ekru',
                  550: 'Kırmızı',
                  600: 'Beyaz',
                  650: 'Krem rengi',
                  653: 'Bej',
                  700: 'Yeşil',
                  701: 'Açık Yeşil',
                  704: 'Su Yeşili',
                  710: 'Haki',
                  750: 'Bordo',
                  752: 'Koyu Bordo',
                  870: 'Taş',
                  925: 'Muhtelif'
                  }
        dataframe['ColorName'] = dataframe['ColorID'].map(colors)
        dataframe.insert(12, 'ColorName', dataframe.pop('ColorName'))

        # ColorID değişkeninin tipini integer yapalım
        dataframe['ColorID'] = dataframe['ColorID'].astype(int)

        # Saat, gün adı bilgilerini içeren yeni sütunlar ekle
        dataframe['Hour'] = dataframe['DateTime'].dt.hour
        dataframe['DayName'] = dataframe['DateTime'].dt.day_name()

        dataframe['Month'] = dataframe['DateTime'].dt.month
        dataframe['MonthName'] = dataframe['DateTime'].dt.month_name()

        # Ürün başına düşen fiyatı hesapla
        dataframe['PricePerUnit'] = dataframe['TotalPrice'] / dataframe['Amount']
        cat_cols, num_cols, cat_but_car = grab_col_names(dataframe)
        # cat_cols'u yeniden tanımlayalım
        cat_cols.append('ProductDescription')
        # num_cols'u yeniden tanımlayalım
        drop_num_cols = ['DateTime', 'ColorID', 'ProductTypeID']
        num_cols = [col for col in num_cols if col not in drop_num_cols]
        # cat_but_car'ı yeniden tanımlayalım
        cat_but_car = [col for col in cat_but_car if col not in 'ProductDescription']
        cat_but_car.extend(['ColorID', 'ProductTypeID', 'DateTime'])
        dataframe.isnull().sum()
        dataframe = dataframe.fillna(0)
        outlier_hours = [23, 0, 8, 1, 2, 7, 6]
        dataframe = dataframe[~dataframe['Hour'].isin(outlier_hours)]
        replace_with_thresholds(dataframe, "Amount")
        replace_with_thresholds(dataframe, "TotalPrice")
        replace_with_thresholds(dataframe, "PricePerUnit")
        return dataframe
def rfm_segment(dataframe, today_date=None):
        if today_date is None:
            today_date = dt.datetime.now()

        df_x = dataframe.groupby('CustomerID').agg({
            'TotalPrice': lambda x: x.sum(),
            'DateTime': lambda x: (today_date - x.max()).days
        })

        df_y = dataframe.groupby(['CustomerID', 'InvoiceNo']).agg({
            'TotalPrice': lambda x: x.sum()
        })

        df_z = df_y.groupby('CustomerID').agg({
            'TotalPrice': lambda x: len(x)
        })

        rfm_table = pd.merge(df_x, df_z, on='CustomerID')

        rfm_table.rename(columns={
            'DateTime': 'Recency',
            'TotalPrice_y': 'Frequency',
            'TotalPrice_x': 'Monetary'
        }, inplace=True)

        return rfm_table
def kmeans_clustering_with_labels(rfm_table, init='k-means++', num_clusters=6):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(rfm_table[['Monetary', 'Recency', 'Frequency']])

    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(scaled_data)

    rfm_table['Cluster'] = kmeans.labels_

    cluster_names = {
        0: 'Bronz Müşteri',
        1: 'Bakır Müşteri',
        2: 'VIP Müşteri',
        3: 'Platin Müşteri',
        4: 'Gümüş Müşteri',
        5: 'Altın Müşteri'
    }
    rfm_table['Cluster_Name'] = rfm_table['Cluster'].map(cluster_names)

    rfm_table.reset_index(inplace=True)

    return kmeans, rfm_table, cluster_names

def marketing_recommendations(segment):
    recommendations = {
        'Bronz Müşteri': [
            'Sadakat programına katılma teklifi sunun.',
            'İndirim kuponları veya promosyonlarla teşvik edin.',
            'Yeni ürünleri denemeleri için özel fırsatlar sunun.'
        ],
        'Bakır Müşteri': [
            'Yüksek değerli ürünleri tanıtmak için özel etkinliklere davet edin.',
            'Sadakat programı avantajları hakkında bilgi verin.',
            'Özel müşteri etkinliklerine katılmaya teşvik edin.'
        ],
        'VIP Müşteri': [
            'Özel koleksiyonları ilk görenler arasında olma avantajı sunun.',
            'Kişisel alışveriş danışmanlığı hizmeti sunun.',
            'VIP müşteri etkinliklerine özel davetler gönderin.'
        ],
        'Platin Müşteri': [
            'Özel sürpriz hediyeler sunun.',
            'Kişisel alışveriş danışmanlığı ve özel sipariş avantajları sağlayın.',
            'VIP etkinliklere özel katılım ve ayrıcalıklar sunun.'
        ],
        'Gümüş Müşteri': [
            'Yenilikçi ürünleri denemeleri için teşvik edin.',
            'Sadakat programı avantajları hakkında bilgi verin.',
            'Özel indirimler ve promosyonlar sunun.'
        ],
        'Altın Müşteri': [
            'Müşteriye özel tasarlanmış ürün veya hizmetler sunun.',
            'Özel etkinliklerde VIP konuk olarak ağırlayın.',
            'Sadakat programında üst seviyelere özel avantajlar sunun.'
        ]
    }

    return recommendations.get(segment, [])

file_path = "Miuul - Project/Mağaza Satış 2023.xlsx"
data3 = import_data3(file_path)
data3 = preprocess_data(data3)
manual_today_date = dt.datetime(2023, 12, 12)
rfm_table = rfm_segment(data3, today_date=manual_today_date)
rfm_table = rfm_table[(rfm_table['Monetary'] > 0)]
rfm_table = rfm_table[(rfm_table['Recency'] > 0)]
rfm_table = rfm_table[(rfm_table['Frequency'] > 0)]
kmeans_model, rfm_table, cluster_names = kmeans_clustering_with_labels(rfm_table, num_clusters=5)

tab_segment.subheader("Müşteri Segmentasyon Sistemi")
tab_segment.markdown("Müşterilerinizi daha iyi anlamak ve özelleştirilmiş pazarlama stratejileri oluşturmak için geliştirdiğimiz "
                     "sistemimize hoş geldiniz! Davranış ve tercihleri analiz ederek müşteri segmentlerinizi belirlememize yardımcı olan "
                     "bu araçla işinizi daha verimli yönetebilirsiniz.")

recency = tab_segment.number_input("Müşteri En Son Kaç Gün Önce Alışveriş Yaptı? ", value=0)
frequency = tab_segment.number_input("Müşteri Son 1 Yılda Kaç Kez Alışveriş Yapmış?", value=0)
monetary = tab_segment.number_input("Müşteri Son 1 Yılda Kaç TL'lik Alışveriş Yapmış?", value=0)

user_data = [[monetary, recency, frequency]]
scaler = StandardScaler()
scaled_user_data = scaler.fit_transform(user_data)
predicted_cluster = kmeans_model.predict(scaled_user_data)[0]
predicted_cluster_name = cluster_names[predicted_cluster]

if tab_segment.button("Müşteri Segmentini Tahmin Et"):
        tab_segment.success(f"Tahmin Edilen Müşteri Segmenti: {predicted_cluster_name}")

# Pazarlama stratejisi önerileri butonu
if tab_segment.button("Pazarlama Stratejisi Önerileri Al"):
    if predicted_cluster_name != "Geçersiz Segment":
        marketing_strategy = marketing_recommendations(predicted_cluster_name)
        if marketing_strategy:
            tab_segment.markdown("**Pazarlama Stratejisi Önerileri:**")
            for strategy in marketing_strategy:
                tab_segment.write(f"- {strategy}")
        else:
            tab_segment.warning("Bu segment için pazarlama stratejisi önerisi bulunamadı.")
    else:
        tab_segment.error("Geçersiz müşteri segmenti.")



####################################
# ÜRÜN ÖNERİ SEKMESİ
####################################

@st.cache_data
def import_data1(file_path):
    # Veri Setinin Import Edilmesi
    data1 = pd.read_excel(file_path)
    return data1
def outlier_thresholds(dataframe, variable):
    # Aykırı değerleri tespit etmek için eşik değerleri hesaplayan fonksiyon
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit
def replace_with_thresholds(dataframe, variable):
    # Aykırı değerleri eşik değerleri ile değiştiren fonksiyon
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

def preprocess_data(dataframe):
    # Veri ön işleme adımlarını içeren fonksiyon
    drop_cols = ['InvoiceTime', 'InvoiceDate', 'OtherPayments']
    dataframe = dataframe.drop(drop_cols, axis=1)

    drop_value = ['ADANA OPTIMUM PANAYIR', 'PANAYIR OPTIMUM', 'URFA PANAYIR', 'BİLİR GİYİM SAN.TİC.A.Ş.']
    dataframe = dataframe[~dataframe['NameSurname'].isin(drop_value)]
    dataframe = dataframe[dataframe['OfficeName'] != 'KURUMSAL MAĞAZA']
    dataframe = dataframe[~dataframe['ProductDescription'].isin(['Bez Büyük Boy Çanta', 'Bez Orta Boy Çanta'])]
    dataframe = dataframe[~dataframe['ProductType'].isin(['ÇORAP', 'Classic Cep Parfümü', 'HEDİYE KARTI', 'MASKE'])]

    colors = {1: 'Siyah', 2: 'Siyah', 50: 'Füme', 53: 'Antrasit', 100: 'Gri', 102: 'Grimel', 103: 'AÇIK GRİ',
              105: 'Orta Gri', 156: 'Somon', 200: 'Lacivert', 201: 'KOYU LACİVERT', 210: 'Marine', 218: 'Parlement',
              300: 'Mavi', 308: 'İndigo', 309: 'Mor', 311: 'A. MAVİ', 312: 'Saks Mavisi', 370: 'Mürdüm', 400: 'KAHVE',
              401: 'A-Kahverengi', 403: 'Kahverengi - Lacivert', 405: 'AÇIK KAHVE', 406: 'Fındık', 408: 'Camel',
              411: 'AÇIK KAHVE', 450: 'ORANJ', 453: 'Yavruağzı', 454: 'Ekru', 550: 'Kırmızı', 600: 'Beyaz',
              650: 'Krem rengi', 653: 'Bej', 700: 'Yeşil', 701: 'Açık Yeşil', 704: 'Su Yeşili', 710: 'Haki',
              750: 'Bordo', 752: 'Koyu Bordo', 870: 'Taş', 925: 'Muhtelif'}
    dataframe['ColorName'] = dataframe['ColorID'].map(colors)
    dataframe.dropna(inplace=True)
    dataframe = dataframe[~(dataframe['Return'] == 1)]
    dataframe = dataframe[dataframe['Amount'] > 0]
    dataframe = dataframe[dataframe['TotalPrice'] > 0]
    replace_with_thresholds(dataframe, "Amount")
    replace_with_thresholds(dataframe, "TotalPrice")
    return dataframe

def create_invoice_product_df(dataframe, id=False):
    # Fatura-Ürün çiftlerini oluşturan fonksiyon
    if id:
        return dataframe.groupby(['InvoiceNo', 'ProductTypeID'])['Amount'].sum().unstack().reset_index().fillna(
            0).set_index('InvoiceNo')
    else:
        return dataframe.groupby(['InvoiceNo', 'ProductType'])['Amount'].sum().unstack().reset_index().fillna(
            0).set_index('InvoiceNo')

def num(x):
    # 0'dan küçük değerleri 0'a, 1'den büyük değerleri 1'e çeviren fonksiyon
    if x <= 0:
        return 0
    if x >= 1:
        return 1

def association_rules_analysis(invoice_product_df):
    # Birliktelik Kurallarının Çıkarılması
    frequent_itemsets = apriori(invoice_product_df, min_support=0.02, use_colnames=True)
    frequent_itemsets.sort_values("support", ascending=False)
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
    return rules

def recommend_products(rules, product_name, top_n=3):
    # Birliktelik kuralları içinde belirli bir ürün için öneri bulan fonksiyon
    sorted_rules = rules.sort_values("lift", ascending=False)
    recommendation_list = []

    for i, product in enumerate(sorted_rules["antecedents"]):
        for j in list(product):
            if j == product_name:
                recommendation_list.append(list(sorted_rules.iloc[i]["consequents"])[0])

    return recommendation_list[:top_n]


# Fonksiyonların çağrılması
file_path = "Miuul - Project/Mağaza Satış 2023.xlsx"  # Veri seti dosya yolu
data1 = import_data1(file_path)
data1 = preprocess_data(data1)
invoice_product_df = create_invoice_product_df(data1)
invoice_product_df = invoice_product_df.applymap(num)
# Birliktelik Kuralları Analizi
rules = association_rules_analysis(invoice_product_df)

# TAB Ürün Önerisi

tab_recommen.image("Miuul - Project/recoo.png")
# Streamlit Arayüzü
tab_recommen.subheader("Ürün Öneri Sistemi")
tab_recommen.markdown("Modern alışveriş dünyasında müşteri deneyimini daha kişisel ve etkileyici hale getirmek için buradayız! "
                      "İyi bir alışveriş deneyimi sunan ürün öneri sistemimiz ile tanışın.")
tab_recommen.markdown("Sistemimiz, müşteri tercihlerini ve alışveriş geçmişini analiz ederek size özel ürün önerileri sunar. "
                      "Böylece, sadece ihtiyaçlarınıza uygun ürünleri keşfetmekle kalmaz, aynı zamanda alışverişinizi daha "
                      "keyifli ve verimli hale getirebilirsiniz.")

# Ürün Kategorisi Seçme
selected_category = tab_recommen.selectbox("Ürün Kategorisi Seçin", data1['ProductType'].unique())

# Ürün Önerilerini Gösterme
if tab_recommen.button("Ürün Önerilerini Getir"):
    recommended_products = recommend_products(rules, selected_category, top_n=3)

    # Benzersiz önerileri al
    unique_recommended_products = list(set(recommended_products))

    # Tekrar eden öğeleri kontrol et ve sadece bir kere göster
    final_recommendations = []
    for product in unique_recommended_products:
        if recommended_products.count(product) > 1:
                final_recommendations.append(product)
        else:
            final_recommendations.append(product)

    tab_recommen.success(f"Seçtiğiniz ürün kategorisi için önerilen ürünler: {', '.join(final_recommendations)}")




































