#####################################################################################


%matplotlib notebook
fig, ax = plt.subplots()
ax.scatter(latents_tsne[:, 0], latents_tsne[:, 1])
ax.set_title('Gene embeddings using Mean pooling')
crs = mplcursors.cursor(ax,highlight=True)
crs.connect("add", lambda sel: sel.annotation.set_text(genes["Gene name"][sel.index]))

plt.show()

#####################################################################################

## Baseline [Subcell]: $Gene2Vec_{base}$ with LR

Gene2Vec = dict()

file_path = f'data/gene2vec_embeddings.txt'
with open(file_path, 'r') as file:
    for line in file:
        
        g_name, g_embed = line.strip().split("	")
        g_embed = [float(value) for value in g_embed.split()] 
        
        Gene2Vec[g_name.strip()] = g_embed

genes_dict = genes_loc.set_index("Gene name")["Y0"].to_dict()

X_train_loc, y_train_loc = [], []
for g in genes_train:
    
    if Gene2Vec.get(g, False) and genes_dict.get(g, False):
        X_train_loc.append(Gene2Vec[g])
        y_train_loc.append(genes_dict[g])
        
X_test_loc, y_test_loc = [], []
for g in genes_test:
    
    if Gene2Vec.get(g, False) and genes_dict.get(g, False):
        X_test_loc.append(Gene2Vec[g])
        y_test_loc.append(genes_dict[g])

print(len(X_test_loc))

scaler = StandardScaler().fit(X_train_loc)

X_train_loc = scaler.transform(X_train_loc)
X_test_loc = scaler.transform(X_test_loc)

classifier = LogisticRegression()
classifier.fit(X_train_loc, y_train_loc)

y_pred_loc = classifier.predict(X_test_loc)

acc = accuracy_score(y_test_loc, y_pred_loc)
f1 = f1_score(y_test_loc, y_pred_loc, average='weighted')


print(f"Accuracy: {acc}")
print(f"F1: {f1}")



#####################################################################################


## Baseline [Subcell] classifier: GeneLLM_Base LR


from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

train_loader, val_loader, test_loader = process_data(genes_loc, 250 , 80)
X_train, y_train, genes_train = getPretrainedEmbeddings(train_loader)
X_test, y_test, genes_test = getPretrainedEmbeddings(test_loader)


scaler = StandardScaler().fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

classifier = LogisticRegression()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')


print(f"Accuracy: {acc}")
print(f"F1: {f1}")

#####################################################################################

def getPretrainedEmbeddings(loader, max_length=250, batch_size=1000, pool ="cls"):
    
    
    model = FineTunedBERT(pool= pool).to("cuda")
    
    print("Get embeddings ...")
    
    embeddings=[]
    labels = []
    genes = []
    genes_dict = genes_phast["Gene name"].to_dict()
    model.eval()
    for batch_input_ids, batch_attention_mask, batch_labels, g_name in tqdm(loader):
        with torch.no_grad():
            
            pooled_embeddings, _, _ = model(batch_input_ids.to("cuda") , batch_attention_mask.to("cuda"))
            embeddings.append(pooled_embeddings.cpu().detach())
            labels.append(batch_labels)
            
            for g in g_name:
                genes.append(genes_dict[int(g)])
    
    
    concat_embeddings = torch.cat(embeddings, dim=0)
    labels = torch.cat(labels, dim=0)
    
    print(concat_embeddings.size())
    
    return concat_embeddings, labels, genes


#####################################################################################

# Baseline: Gene2Vec  #

Gene2Vec = dict()

file_path = f'data/gene2vec_embeddings.txt'
with open(file_path, 'r') as file:
    for line in file:
        
        g_name, g_embed = line.strip().split("	")
        g_embed = [float(value) for value in g_embed.split()] 
        
        Gene2Vec[g_name.strip()] = g_embed

genes_dict = genes_phast.set_index("Gene name")["Conservation"].to_dict()

X_train_vec, y_train_vec = [], []
for g in genes_train:
    
    if Gene2Vec.get(g, False):
        X_train_vec.append(Gene2Vec[g])
        y_train_vec.append(genes_dict[g])
        
X_test_vec, y_test_vec = [], []
for g in genes_test:
    
    if Gene2Vec.get(g, False):
        X_test_vec.append(Gene2Vec[g])
        y_test_vec.append(genes_dict[g])


regressor = LinearRegression()
regressor.fit(X_train_vec, y_train_vec)

y_pred_vec = regressor.predict(X_test_vec)

mse_vec = mean_squared_error(y_test_vec, y_pred_vec)
r2_vec = r2_score(y_test_vec, y_pred_vec)

print(f"Mean Squared Error: {mse_vec}")
print(f"corr: {spearmanr(y_test_vec, y_pred_vec)[0]}")


#####################################################################################
def plot_latent(latents, epoch, validation_type="train"):
    
    tsne = TSNE(n_components=2)
    scaler = StandardScaler()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        latents_tsne = tsne.fit_transform(latents)
    
    
    plt.scatter(latents_tsne[:, 0], latents_tsne[:, 1])
    
    plt.title(f'Epoch {epoch}')
    plt.savefig(f"saved-figures/latent/{validation_type}/latent_{epoch}.png")
    plt.close()


def train(loader, model, optimizer):
    
    
    train_loss = 0
    latents  = []
    total_preds = []
    total_labels = []
    total_corr = 0
    MSELoss = nn.MSELoss()
    
    model.train()
    for batch in tqdm(loader): 

        batch_inputs, batch_masks, batch_labels =  batch[0].to("cuda") , batch[1].to("cuda"), batch[2].to("cuda")

        
        pooled_embeddings, _ , preds = model(batch_inputs, batch_masks)

        preds = preds.squeeze().float()
        labels = batch_labels.squeeze().float()
        loss = MSELoss(preds, labels)
        
        total_corr += spearmanr(preds.cpu().detach(), labels.cpu().detach())[0]
        train_loss += loss.item()

        
        #Aggregation
        pooled_embeddings = torch.tensor(pooled_embeddings.cpu().detach().numpy())
        latents.append(pooled_embeddings) 
        total_preds.extend(preds.cpu().detach())
        total_labels.extend(labels.cpu().detach())

            
        model.zero_grad()
        loss.backward()
        optimizer.step()
        
    train_loss /= len(loader)
    total_corr /= len(loader)
    latents = torch.cat(latents, dim=0)

#     print(f"total_corr:{spearmanr(total_preds, total_labels)[0]}")
#     print(f"batch_corr:{total_corr}")
    
                
    return model, train_loss, total_corr,  latents




def validation (loader, model):
    
    val_loss = 0
    total_corr = 0
    MSELoss = nn.MSELoss()
    
    model.eval()
    for batch in tqdm(loader): 

        batch_inputs, batch_masks, batch_labels =  batch[0].to("cuda") , batch[1].to("cuda"), batch[2].to("cuda")

        
        pooled_embeddings, _ , preds = model(batch_inputs, batch_masks)

        preds = preds.squeeze().float()
        labels = batch_labels.squeeze().float()
        loss = MSELoss(preds, labels)
        
        total_corr += spearmanr(preds.cpu().detach(), labels.cpu().detach())[0]
        
        val_loss += loss.item()

    
    val_loss /= len(loader)
    total_corr /= len(loader)
    

                
    return model, val_loss, total_corr


def test(loader, model):
    
    test_loss = 0
    total_corr = 0
    MSELoss = nn.MSELoss()
    latents = []
    model.eval()
    
    for batch in tqdm(loader): 

        batch_inputs, batch_masks, batch_labels =  batch[0].to("cuda") , batch[1].to("cuda"), batch[2].to("cuda")

        
        pooled_embeddings, _ , preds = model(batch_inputs, batch_masks)

        preds = preds.squeeze().float()
        labels = batch_labels.squeeze().float()
        loss = MSELoss(preds, labels)
        
        total_corr += spearmanr(preds.cpu().detach(), labels.cpu().detach())[0]
        
        test_loss += loss.item()

        pooled_embeddings = torch.tensor(pooled_embeddings.cpu().detach().numpy())
        latents.append(pooled_embeddings)
        
        
    test_loss /= len(loader)
    total_corr /= len(loader)
    latents = torch.cat(latents, dim=0)

                
    return model, test_loss,total_corr, latents

def process_data(genes, max_length, batch_size,
                 gene2vec_flag = False, model_name= "bert-base-cased"):
    
    tokenizer = BertTokenizerFast.from_pretrained(model_name)
    
    sentences, labels, g_names = genes["Summary"].tolist() , genes["Conservation"].tolist(), genes.index.tolist()
    
    tokens = tokenizer.batch_encode_plus(sentences, max_length = max_length, padding="max_length",
                                         truncation=True)

    data = {'input_ids': tokens["input_ids"],
            'token_type_ids': tokens["token_type_ids"],
            'attention_mask': tokens["attention_mask"],
            "labels": labels,
            "g_name": g_names
           }
    
    tokens_df = pd.DataFrame(data)
    #############################################
    if gene2vec_flag:
        print("Adding Gene2Vec data ...")
        
        Gene2Vec = dict()

        file_path = f'data/gene2vec_embeddings.txt'
        with open(file_path, 'r') as file:
            for line in file:

                g_name, g_embed = line.strip().split("	")
                g_embed = [float(value) for value in g_embed.split()] 

                Gene2Vec[g_name.strip()] = g_embed
        
        #This will not work
        tokens_df["gene2vec"] = tokens_df["g_name"].apply(lambda name: Gene2Vec[name])
    #############################################
    
    
    train_tokens, test_tokens = train_test_split(tokens_df, test_size=0.15,
                                                 random_state=42)
    
    train_tokens, val_tokens = train_test_split(train_tokens,test_size=0.20,
                                                random_state=42)

    train_tokens = train_tokens.reset_index(drop=True)
    val_tokens = val_tokens.reset_index(drop=True)
    test_tokens = test_tokens.reset_index(drop=True)
    
    train_dataset = TensorDataset(torch.tensor(train_tokens["input_ids"].tolist()),
                                  torch.tensor(train_tokens["attention_mask"].tolist()),
                                  torch.tensor(train_tokens["gene2vec"]),
                                  torch.tensor(train_tokens["labels"]),
                                  torch.tensor(train_tokens["g_name"])
                                 )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    
    
    
    val_dataset = TensorDataset(torch.tensor(val_tokens["input_ids"].tolist()) ,
                                torch.tensor(val_tokens["attention_mask"].tolist()),
                                torch.tensor(val_tokens["gene2vec"]),
                                torch.tensor(val_tokens["labels"]),
                                torch.tensor(val_tokens["g_name"])
                               )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    
    
    test_dataset = TensorDataset(torch.tensor(test_tokens["input_ids"].tolist()),
                                 torch.tensor(test_tokens["attention_mask"].tolist()),
                                 torch.tensor(test_tokens["gene2vec"]),
                                 torch.tensor(test_tokens["labels"]),
                                 torch.tensor(test_tokens["g_name"])
                                )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    
    return train_loader, val_loader, test_loader

def trainer(epochs, genes, lr =5e-5, pool="cls",
            max_length= 100, batch_size =100, model_name= "bert-base-cased"):
    
    
    model = FineTunedBERT(pool= pool, task = "reg").to("cuda")
    
    optimizer = AdamW(model.parameters(), lr=lr, eps=1e-8)
    
    train_loader, val_loader, test_loader = process_data(genes, max_length,
                                                         batch_size, gene2vec_flag = gene2vec_flag,
                                                         model_name = model_name)
    
    for epoch in range(epochs):
        start_time = time.time()
        
        print(f"Epoch {epoch+1} of {epochs}")
        print("-------------------------------")
        
        print("Training ...")
        model, train_loss, train_corr, latents = train(train_loader, model, optimizer)
        plot_latent(latents, epoch, validation_type="train")
        
        print("Validation ...")
        model, val_loss, val_corr  = validation (val_loader, model)
        
        print("Testing ...")
        model, test_loss, test_corr, _ = test (test_loader, model)
        
        print(f'\tET: {round(time.time() - start_time,2)} Seconds')
        print(f'\tTrain Loss: {round(train_loss,4)}, corrcoef: {round(train_corr,4)}')
        print(f'\tVal Loss: {round(val_loss,4)}, corrcoef: {round(val_corr,4)}')
        print(f'\tTest Loss: {round(test_loss,4)}, corrcoef: {round(test_corr,4)}')
