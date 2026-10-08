import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

RED, DARK = "#E50914", "#221F1F"
sns.set_theme(style="whitegrid"); plt.rcParams.update({"axes.titleweight":"bold","axes.titlesize":13})

df = pd.read_csv("data/netflix_titles.csv")
df["director"]=df.director.fillna("Unknown"); df["cast"]=df.cast.fillna("Unknown")
df["country"]=df.country.fillna("Unknown"); df["rating"]=df.rating.fillna("Unknown")
df["date_added"]=pd.to_datetime(df.date_added.str.strip(), errors="coerce")
df["year_added"]=df.date_added.dt.year
df["dur_num"]=df.duration.str.extract(r"(\d+)").astype(float)
print("Rows:",len(df),"| Movies:",(df.type=="Movie").sum(),"| TV:",(df.type=="TV Show").sum())

def top(col, n=10):
    c=Counter(x.strip() for s in df[col] for x in s.split(",") if x.strip() not in ("Unknown",""))
    return pd.Series(dict(c.most_common(n)))

# 1 type split
fig,ax=plt.subplots(figsize=(5,5))
vc=df.type.value_counts(); ax.pie(vc,labels=vc.index,autopct="%1.0f%%",colors=[RED,DARK],startangle=90,textprops={"color":"w","weight":"bold","fontsize":13},wedgeprops={"edgecolor":"w"})
for t in ax.texts[:2]: t.set_color("black")
ax.set_title("Movies vs TV Shows"); plt.savefig("out/01_type_split.png",dpi=160,bbox_inches="tight"); plt.close()

# 2 growth
g=df.groupby(["year_added","type"]).size().unstack(fill_value=0); g=g[(g.index>=2012)&(g.index<=2020)]; g.index=g.index.astype(int)
fig,ax=plt.subplots(figsize=(8,4.5)); g.plot(ax=ax,color={"Movie":RED,"TV Show":DARK},lw=3,marker="o")
ax.set_title("Content Added to Netflix per Year"); ax.set_xticks(g.index); ax.set_xlabel(""); ax.set_ylabel("Titles added")
plt.savefig("out/02_growth.png",dpi=160,bbox_inches="tight"); plt.close()

# 3 genres
gen=top("listed_in",10)
fig,ax=plt.subplots(figsize=(8,4.5)); sns.barplot(x=gen.values,y=gen.index,color=RED,ax=ax); ax.set_title("Top 10 Genres"); ax.set_xlabel("Titles")
plt.savefig("out/03_genres.png",dpi=160,bbox_inches="tight"); plt.close()

# 4 countries
cty=top("country",10)
fig,ax=plt.subplots(figsize=(8,4.5)); sns.barplot(x=cty.values,y=cty.index,color=DARK,ax=ax); ax.set_title("Top 10 Producing Countries"); ax.set_xlabel("Titles")
plt.savefig("out/04_countries.png",dpi=160,bbox_inches="tight"); plt.close()

# 5 ratings
r=df.rating.value_counts().head(8)
fig,ax=plt.subplots(figsize=(8,4.5)); sns.barplot(x=r.index,y=r.values,color=RED,ax=ax); ax.set_title("Maturity Ratings"); ax.set_ylabel("Titles")
plt.savefig("out/05_ratings.png",dpi=160,bbox_inches="tight"); plt.close()

# 6 duration
mv=df[df.type=="Movie"].dur_num.dropna(); tv=df[df.type=="TV Show"].dur_num.dropna()
fig,ax=plt.subplots(1,2,figsize=(10,4))
sns.histplot(mv,bins=40,color=RED,ax=ax[0]); ax[0].set_title(f"Movie Length (median {mv.median():.0f} min)"); ax[0].set_xlabel("Minutes")
sns.countplot(x=tv.astype(int),color=DARK,ax=ax[1]); ax[1].set_title("TV Show Seasons"); ax[1].set_xlabel("Seasons")
plt.tight_layout(); plt.savefig("out/06_duration.png",dpi=160,bbox_inches="tight"); plt.close()

# Recommender
df["soup"]=(df.listed_in.str.replace(","," ")+" ")*3+df.description+" "+df.cast.str.replace(" ","").str.replace(","," ")+" "+df.director.str.replace(" ","")
tf=TfidfVectorizer(stop_words="english",ngram_range=(1,2),min_df=2,max_features=60000)
X=tf.fit_transform(df.soup); idx=pd.Series(df.index,index=df.title.str.lower()).drop_duplicates()
def rec(t,n=5):
    i=idx[t.lower()]; s=linear_kernel(X[i],X).ravel(); s[i]=-1; o=s.argsort()[::-1][:n]
    return df.title.iloc[o].tolist(), s[o]
demos=[t for t in ["Stranger Things","Narcos","The Crown","Black Mirror"] if t.lower() in idx.index]
print("Demos:",demos)
fig,axs=plt.subplots(1,len(demos),figsize=(4.2*len(demos),3.8))
axs=np.atleast_1d(axs); recs={}
for a,t in zip(axs,demos):
    names,sc=rec(t); recs[t]=names; print(t,"->",list(zip(names,sc.round(2))))
    sns.barplot(x=sc,y=[n[:28] for n in names],color=RED,ax=a); a.set_title(f"Because you watched\n{t}",fontsize=11); a.set_xlabel("Cosine similarity")
plt.tight_layout(); plt.savefig("out/07_recommender.png",dpi=160,bbox_inches="tight"); plt.close()

# Classifier: predict audience group (Kids / Teens / Adults) from description + genres
grp={"TV-Y":"Kids","TV-Y7":"Kids","TV-Y7-FV":"Kids","TV-G":"Kids","G":"Kids","TV-PG":"Kids","PG":"Kids","PG-13":"Teens","TV-14":"Teens","TV-MA":"Adults","R":"Adults","NC-17":"Adults"}
d2=df[df.rating.isin(grp)].copy(); d2["y"]=d2.rating.map(grp)
d2["txt"]=d2.description+" "+d2.listed_in.str.replace("TV Shows","").str.replace("Movies","").str.replace(","," ")
Xtr,Xte,ytr,yte=train_test_split(d2.txt,d2.y,test_size=.2,random_state=42,stratify=d2.y)
base=yte.value_counts(normalize=True).max(); print("Majority baseline: %.3f"%base)
v=TfidfVectorizer(stop_words="english",ngram_range=(1,2),min_df=2,sublinear_tf=True); A=v.fit_transform(Xtr); B=v.transform(Xte)
models={"Naive Bayes":MultinomialNB(),"Linear SVM":LinearSVC(class_weight="balanced"),"Logistic Reg.":LogisticRegression(max_iter=2000,class_weight="balanced")}
res={}
for n,m in models.items():
    m.fit(A,ytr); p=m.predict(B); res[n]=(accuracy_score(yte,p),f1_score(yte,p,average="macro"),p); print(n,"acc %.3f macroF1 %.3f"%res[n][:2])
best=max(res,key=lambda k:res[k][1]); print("Best:",best)
labs=["Kids","Teens","Adults"]
fig,ax=plt.subplots(1,2,figsize=(11,4.3))
rd=pd.DataFrame({k:{"Accuracy":v_[0],"Macro F1":v_[1]} for k,v_ in res.items()}).T
rd.plot.bar(ax=ax[0],color=[RED,DARK],rot=0); ax[0].set_ylim(.3,1); ax[0].axhline(base,ls="--",c="gray"); ax[0].text(2.4,base+.01,"baseline",ha="right",color="gray"); ax[0].set_title("Model Comparison"); ax[0].legend(loc="upper left")
for c in ax[0].containers: ax[0].bar_label(c,fmt="%.2f",fontsize=9)
ConfusionMatrixDisplay(confusion_matrix(yte,res[best][2],labels=labs),display_labels=labs).plot(ax=ax[1],cmap="Reds",colorbar=False); ax[1].set_title(f"Confusion Matrix - {best}"); ax[1].grid(False)
plt.tight_layout(); plt.savefig("out/08_model.png",dpi=160,bbox_inches="tight"); plt.close()
BEST=(best,res[best][0],res[best][1],yte,res[best][2],labs)

# LinkedIn dashboard
fig=plt.figure(figsize=(18,10),facecolor="white"); gs=fig.add_gridspec(2,3,hspace=.45,wspace=.75,top=.86)
fig.suptitle("NETFLIX CONTENT ANALYSIS & RECOMMENDER  |  Python · pandas · scikit-learn",fontsize=18,fontweight="bold",color=RED,y=.97)
a=fig.add_subplot(gs[0,0]); g.plot(ax=a,color={"Movie":RED,"TV Show":DARK},lw=3); a.set_title("Titles Added per Year"); a.set_xlabel(""); a.set_xticks(g.index[::2]); a.legend(title="",loc="upper left")
a=fig.add_subplot(gs[0,1]); sns.barplot(x=gen.values,y=gen.index,color=RED,ax=a); a.set_title("Top Genres"); a.set_xlabel(""); a.set_ylabel("")
a=fig.add_subplot(gs[0,2]); sns.barplot(x=cty.values,y=cty.index,color=DARK,ax=a); a.set_title("Top Countries"); a.set_xlabel(""); a.set_ylabel("")
a=fig.add_subplot(gs[1,0]); sns.barplot(x=r.index,y=r.values,color=RED,ax=a); a.set_title("Maturity Ratings"); a.set_ylabel(""); a.set_xlabel(""); a.tick_params(axis="x",labelsize=9)
a=fig.add_subplot(gs[1,1]); n0=demos[0]; names,sc=rec(n0); sns.barplot(x=sc,y=[x[:26] for x in names],color=DARK,ax=a); a.set_title(f"Recommender: '{n0}'"); a.set_ylabel(""); a.set_xlabel("Cosine similarity")
a=fig.add_subplot(gs[1,2]); ConfusionMatrixDisplay(confusion_matrix(yte,res[best][2],labels=labs),display_labels=labs).plot(ax=a,cmap="Reds",colorbar=False)
a.set_title(f"Audience Classifier (Kids/Teens/Adults)\nAcc {res[best][0]:.0%} vs {base:.0%} baseline | F1 {res[best][1]:.2f}"); a.grid(False)
plt.savefig("out/00_LINKEDIN_dashboard.png",dpi=160,bbox_inches="tight"); plt.close()

# key stats
print("Peak year:",g.sum(axis=1).idxmax(),int(g.sum(axis=1).max()))
print("US share: %.0f%%"%(100*df.country.str.contains("United States").mean()))
print("TV-MA share: %.0f%%"%(100*(df.rating=="TV-MA").mean()))
print("Median movie min:",mv.median(),"| TV 1 season share: %.0f%%"%(100*(tv==1).mean()))
print("Top genre:",gen.index[0],gen.iloc[0],"| top country:",cty.index[0],cty.iloc[0])
