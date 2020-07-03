        filename = "ratings_Patio_Lawn_and_Garden.csv"
        with open(os.path.join("..", "data", filename), "rb") as f:
            ratings = pd.read_csv(f,names=("user","item","rating","timestamp"))
        X, user_mapper, item_mapper, user_inverse_mapper, item_inverse_mapper, user_ind, item_ind = utils.create_user_item_matrix(ratings)
        X_binary = X != 0

        grill_brush = "B00CFM0P7Y"
        grill_brush_ind = item_mapper[grill_brush]
        grill_brush_vec = X[:,grill_brush_ind]

        print(url_amazon % grill_brush)


        # Euclidean distance
        X_t = X.transpose()
        grill_brush_vec_t = grill_brush_vec.transpose()
        neigh = NearestNeighbors(n_neighbors=6)
        neigh.fit(X_t)
        neighbour_index = neigh.kneighbors(grill_brush_vec_t,return_distance=False)
        # print the similar neighbors
        print("Five most similar items using Euclidean Distance:")
        for i in range(6):
            index = neighbour_index[0,i]
            if index != grill_brush_ind:
                item_id = item_inverse_mapper[index]
                print(url_amazon % item_id)
        
        
        Normalized Euclidean distance
        # normalize X first
        normalized_X = normalize(X, axis=0)
        X_t = normalized_X.transpose()
        grill_brush_vec_t = grill_brush_vec.transpose()
        neigh = NearestNeighbors(n_neighbors=6)
        neigh.fit(X_t)
        neighbour_index = neigh.kneighbors(grill_brush_vec_t,return_distance=False)
        # print the similar neighbors
        print("Five most similar items using Normalized Euclidean Distance:")
        for i in range(6):
            index = neighbour_index[0,i]
            if index != grill_brush_ind:
                item_id = item_inverse_mapper[index]
                print(url_amazon % item_id)
        
        
        # Cosine similarity
        X_t = X.transpose()
        grill_brush_vec_t = grill_brush_vec.transpose()
        neigh = NearestNeighbors(n_neighbors=6, metric='cosine')
        neigh.fit(X_t)
        neighbour_index = neigh.kneighbors(grill_brush_vec_t,return_distance=False)
        # print the similar neighbors
        print("Five most similar items using Cosine Similarity:")
        for i in range(6):
            index = neighbour_index[0,i]
            if index != grill_brush_ind:
                item_id = item_inverse_mapper[index]
                print(url_amazon % item_id)
