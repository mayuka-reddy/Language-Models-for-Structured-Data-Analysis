#!/usr/bin/env python3
"""
Kaggle Dataset Import, Cleaning & Processing Script
Downloads Brazilian E-Commerce (Olist) dataset from Kaggle and prepares it for NL-to-SQL training.

Usage:
    python scripts/prepare_dataset.py
    
Requirements:
    - Kaggle API credentials (~/.kaggle/kaggle.json)
    - Or manual download from: https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce
    
Output:
    - data/raw/: Original CSV files from Kaggle
    - data/processed/: Cleaned and processed data
    - data/database/: SQLite database ready for queries
    - data/reports/: Data quality reports
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import sqlite3
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class DatasetPreparer:
    """Complete dataset preparation pipeline."""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        self.database_dir = self.data_dir / "database"
        self.reports_dir = self.data_dir / "reports"
        
        self.setup_directories()
        
        # Dataset info
        self.dataset_name = "olistbr/brazilian-ecommerce"
        self.tables = {}
        self.stats = {}
    
    def setup_directories(self):
        """Create directory structure."""
        for dir_path in [self.raw_dir, self.processed_dir, self.database_dir, self.reports_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        print(f"📁 Data directory: {self.data_dir.absolute()}")
    
    def download_from_kaggle(self):
        """Download dataset from Kaggle using API."""
        print("\n" + "="*70)
        print("📥 DOWNLOADING DATASET FROM KAGGLE")
        print("="*70)
        
        try:
            import kaggle
            print(f"✅ Kaggle API found")
            print(f"📦 Downloading: {self.dataset_name}")
            
            # Download dataset
            kaggle.api.dataset_download_files(
                self.dataset_name,
                path=self.raw_dir,
                unzip=True
            )
            
            print(f"✅ Dataset downloaded to: {self.raw_dir}")
            return True
            
        except ImportError:
            print("❌ Kaggle API not installed")
            print("\n📋 To install:")
            print("   pip install kaggle")
            print("\n📋 Setup Kaggle API:")
            print("   1. Go to https://www.kaggle.com/account")
            print("   2. Create API token (downloads kaggle.json)")
            print("   3. Place in ~/.kaggle/kaggle.json")
            print("   4. chmod 600 ~/.kaggle/kaggle.json")
            return False
            
        except Exception as e:
            print(f"❌ Error downloading: {e}")
            print("\n📋 Manual download:")
            print(f"   1. Visit: https://www.kaggle.com/datasets/{self.dataset_name}")
            print(f"   2. Download and extract to: {self.raw_dir}")
            return False
    
    def load_raw_data(self):
        """Load raw CSV files."""
        print("\n" + "="*70)
        print("📊 LOADING RAW DATA")
        print("="*70)
        
        # Expected files
        expected_files = {
            'customers': 'olist_customers_dataset.csv',
            'orders': 'olist_orders_dataset.csv',
            'order_items': 'olist_order_items_dataset.csv',
            'products': 'olist_products_dataset.csv',
            'order_payments': 'olist_order_payments_dataset.csv',
            'sellers': 'olist_sellers_dataset.csv',
            'geolocation': 'olist_geolocation_dataset.csv',
            'order_reviews': 'olist_order_reviews_dataset.csv',
            'product_category_translation': 'product_category_name_translation.csv'
        }
        
        loaded_count = 0
        for table_name, filename in expected_files.items():
            file_path = self.raw_dir / filename
            
            if file_path.exists():
                try:
                    df = pd.read_csv(file_path)
                    self.tables[table_name] = df
                    print(f"   ✅ {table_name}: {len(df):,} rows, {len(df.columns)} columns")
                    loaded_count += 1
                except Exception as e:
                    print(f"   ❌ Error loading {filename}: {e}")
            else:
                print(f"   ⚠️  {filename} not found")
        
        if loaded_count == 0:
            print("\n❌ No data files found!")
            print("📋 Please download the dataset first")
            return False
        
        print(f"\n✅ Loaded {loaded_count}/{len(expected_files)} tables")
        return True
    
    def clean_data(self):
        """Clean and process data."""
        print("\n" + "="*70)
        print("🧹 CLEANING DATA")
        print("="*70)
        
        for table_name, df in self.tables.items():
            print(f"\n📋 Cleaning: {table_name}")
            
            # Store original stats
            original_rows = len(df)
            original_nulls = df.isnull().sum().sum()
            
            # 1. Remove duplicates
            df = df.drop_duplicates()
            duplicates_removed = original_rows - len(df)
            
            # 2. Handle missing values
            # For numeric columns: fill with median
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if df[col].isnull().any():
                    df[col].fillna(df[col].median(), inplace=True)
            
            # For categorical columns: fill with mode or 'unknown'
            categorical_cols = df.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                if df[col].isnull().any():
                    mode_val = df[col].mode()
                    if len(mode_val) > 0:
                        df[col].fillna(mode_val[0], inplace=True)
                    else:
                        df[col].fillna('unknown', inplace=True)
            
            # 3. Convert date columns
            date_columns = [col for col in df.columns if 'date' in col.lower() or 'timestamp' in col.lower()]
            for col in date_columns:
                try:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                except:
                    pass
            
            # 4. Standardize text columns
            text_cols = df.select_dtypes(include=['object']).columns
            for col in text_cols:
                if col not in date_columns:
                    df[col] = df[col].str.strip()
                    df[col] = df[col].str.lower()
            
            # Update table
            self.tables[table_name] = df
            
            # Store stats
            self.stats[table_name] = {
                'original_rows': original_rows,
                'final_rows': len(df),
                'duplicates_removed': duplicates_removed,
                'original_nulls': original_nulls,
                'final_nulls': df.isnull().sum().sum(),
                'columns': len(df.columns)
            }
            
            print(f"   ✅ Rows: {original_rows:,} → {len(df):,} (-{duplicates_removed})")
            print(f"   ✅ Nulls: {original_nulls:,} → {df.isnull().sum().sum():,}")
    
    def save_processed_data(self):
        """Save cleaned data."""
        print("\n" + "="*70)
        print("💾 SAVING PROCESSED DATA")
        print("="*70)
        
        for table_name, df in self.tables.items():
            output_file = self.processed_dir / f"{table_name}.csv"
            df.to_csv(output_file, index=False)
            print(f"   ✅ Saved: {table_name}.csv ({len(df):,} rows)")
        
        # Save stats
        stats_file = self.reports_dir / "data_cleaning_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(self.stats, f, indent=2)
        print(f"\n   ✅ Stats saved: data_cleaning_stats.json")
    
    def create_sqlite_database(self):
        """Create SQLite database from processed data."""
        print("\n" + "="*70)
        print("🗄️  CREATING SQLITE DATABASE")
        print("="*70)
        
        db_path = self.database_dir / "ecommerce.db"
        
        # Remove existing database
        if db_path.exists():
            db_path.unlink()
        
        # Create connection
        conn = sqlite3.connect(db_path)
        
        # Load main tables
        main_tables = ['customers', 'orders', 'order_items', 'products', 'order_payments']
        
        for table_name in main_tables:
            if table_name in self.tables:
                df = self.tables[table_name]
                df.to_sql(table_name, conn, index=False, if_exists='replace')
                print(f"   ✅ Created table: {table_name} ({len(df):,} rows)")
        
        # Create indexes for better query performance
        print("\n   📊 Creating indexes...")
        cursor = conn.cursor()
        
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_customers_id ON customers(customer_id)",
            "CREATE INDEX IF NOT EXISTS idx_orders_id ON orders(order_id)",
            "CREATE INDEX IF NOT EXISTS idx_orders_customer ON orders(customer_id)",
            "CREATE INDEX IF NOT EXISTS idx_order_items_order ON order_items(order_id)",
            "CREATE INDEX IF NOT EXISTS idx_order_items_product ON order_items(product_id)",
            "CREATE INDEX IF NOT EXISTS idx_products_id ON products(product_id)",
            "CREATE INDEX IF NOT EXISTS idx_payments_order ON order_payments(order_id)"
        ]
        
        for idx_sql in indexes:
            cursor.execute(idx_sql)
        
        conn.commit()
        conn.close()
        
        print(f"   ✅ Indexes created")
        print(f"\n✅ Database created: {db_path}")
        print(f"   Size: {db_path.stat().st_size / 1024 / 1024:.2f} MB")
    
    def generate_data_report(self):
        """Generate comprehensive data quality report."""
        print("\n" + "="*70)
        print("📝 GENERATING DATA REPORT")
        print("="*70)
        
        report = f"""# Data Preparation Report
## Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

### Dataset Information
- **Source**: Brazilian E-Commerce Public Dataset (Olist)
- **Kaggle**: https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce
- **Tables Processed**: {len(self.tables)}

### Data Cleaning Summary

| Table | Original Rows | Final Rows | Duplicates Removed | Nulls Cleaned | Columns |
|-------|---------------|------------|-------------------|---------------|---------|
"""
        
        for table_name, stats in self.stats.items():
            report += f"| {table_name} | {stats['original_rows']:,} | {stats['final_rows']:,} | {stats['duplicates_removed']:,} | {stats['original_nulls']:,} → {stats['final_nulls']:,} | {stats['columns']} |\n"
        
        report += f"""
### Main Tables Schema

#### customers
- customer_id (PK)
- customer_unique_id
- customer_zip_code_prefix
- customer_city
- customer_state

#### orders
- order_id (PK)
- customer_id (FK)
- order_status
- order_purchase_timestamp
- order_delivered_customer_date
- order_estimated_delivery_date

#### order_items
- order_id (FK)
- order_item_id
- product_id (FK)
- seller_id (FK)
- price
- freight_value

#### products
- product_id (PK)
- product_category_name
- product_weight_g
- product_length_cm
- product_height_cm
- product_width_cm

#### order_payments
- order_id (FK)
- payment_sequential
- payment_type
- payment_installments
- payment_value

### Files Generated

- **Raw Data**: `data/raw/*.csv` (original Kaggle files)
- **Processed Data**: `data/processed/*.csv` (cleaned data)
- **Database**: `data/database/ecommerce.db` (SQLite database)
- **Reports**: `data/reports/data_cleaning_stats.json`

### Next Steps

1. **Use in Training**:
   ```bash
   python scripts/train_all_techniques.py
   ```

2. **Query Database**:
   ```python
   import sqlite3
   conn = sqlite3.connect('data/database/ecommerce.db')
   df = pd.read_sql_query("SELECT * FROM customers LIMIT 5", conn)
   ```

3. **Load Processed Data**:
   ```python
   import pandas as pd
   customers = pd.read_csv('data/processed/customers.csv')
   ```

### Data Quality Metrics

- **Total Records**: {sum(stats['final_rows'] for stats in self.stats.values()):,}
- **Total Duplicates Removed**: {sum(stats['duplicates_removed'] for stats in self.stats.values()):,}
- **Total Nulls Cleaned**: {sum(stats['original_nulls'] for stats in self.stats.values()):,}
- **Database Size**: {(self.database_dir / 'ecommerce.db').stat().st_size / 1024 / 1024:.2f} MB

### Ready for NL-to-SQL Training! ✅
"""
        
        report_file = self.reports_dir / "data_preparation_report.md"
        with open(report_file, 'w') as f:
            f.write(report)
        
        print(f"✅ Report saved: {report_file}")
        
        # Also print summary
        print("\n" + "="*70)
        print("📊 DATA PREPARATION SUMMARY")
        print("="*70)
        print(f"✅ Tables processed: {len(self.tables)}")
        print(f"✅ Total records: {sum(stats['final_rows'] for stats in self.stats.values()):,}")
        print(f"✅ Database created: {self.database_dir / 'ecommerce.db'}")
        print(f"✅ Ready for training!")
    
    def run_complete_pipeline(self):
        """Run the complete data preparation pipeline."""
        start_time = datetime.now()
        
        print("\n" + "="*70)
        print("🚀 STARTING DATA PREPARATION PIPELINE")
        print("="*70)
        print(f"📅 Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Step 1: Download (if needed)
        if not list(self.raw_dir.glob('*.csv')):
            print("\n📥 No raw data found. Attempting download...")
            if not self.download_from_kaggle():
                print("\n⚠️  Please download dataset manually and place in data/raw/")
                return False
        else:
            print(f"\n✅ Raw data found in: {self.raw_dir}")
        
        # Step 2: Load
        if not self.load_raw_data():
            return False
        
        # Step 3: Clean
        self.clean_data()
        
        # Step 4: Save
        self.save_processed_data()
        
        # Step 5: Create database
        self.create_sqlite_database()
        
        # Step 6: Generate report
        self.generate_data_report()
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print("\n" + "="*70)
        print("🎉 DATA PREPARATION COMPLETE!")
        print("="*70)
        print(f"⏱️  Duration: {duration:.2f} seconds")
        print(f"📁 Data directory: {self.data_dir.absolute()}")
        print("\n🎯 Next Steps:")
        print("   1. Review: data/reports/data_preparation_report.md")
        print("   2. Train models: python scripts/train_all_techniques.py")
        print("   3. Launch frontend: python launch_frontend.py")
        
        return True


def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Prepare Brazilian E-Commerce dataset for NL-to-SQL training',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data',
        help='Data directory (default: data)'
    )
    
    args = parser.parse_args()
    
    try:
        preparer = DatasetPreparer(data_dir=args.data_dir)
        success = preparer.run_complete_pipeline()
        return 0 if success else 1
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Preparation interrupted by user")
        return 1
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())