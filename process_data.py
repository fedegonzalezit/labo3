import polars as pl

print("Reading files...")
# Read the files with Polars
sell_in = pl.read_csv(
    'sell-in.txt',
    separator='\t',
    dtypes={
        'periodo': pl.String,
        'customer_id': pl.String,
        'product_id': pl.String,
        'plan_precios_cuidados': pl.Int32,
        'cust_request_qty': pl.Float64,
        'cust_request_tn': pl.Float64,
        'tn': pl.Float64
    }
)

stocks = pl.read_csv(
    'tb_stocks.txt',
    separator='\t',
    dtypes={
        'periodo': pl.String,
        'product_id': pl.String,
        'stock_final': pl.Float64
    }
)

productos = pl.read_csv(
    'tb_productos.txt',
    separator='\t',
    dtypes={
        'cat1': pl.String,
        'cat2': pl.String,
        'cat3': pl.String,
        'brand': pl.String,
        'sku_size': pl.String,
        'product_id': pl.String
    }
)

print("Writing to parquet...")
sell_in.write_parquet('sell_in.parquet')
stocks.write_parquet('stocks.parquet')
productos.write_parquet('productos.parquet')

print("Done! Files saved as parquet:")
print("- sell_in.parquet")
print("- stocks.parquet")
print("- productos.parquet") 