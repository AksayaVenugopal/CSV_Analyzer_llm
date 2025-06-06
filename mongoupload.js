const fs = require('fs');
const path = require('path');
const csv = require('csv-parser');
const { MongoClient } = require('mongodb');

class CSVMongoUploader {
    constructor(mongoUri, dbName) {
        this.mongoUri = mongoUri;
        this.dbName = dbName;
        this.client = null;
        this.db = null;
    }

    // Connect to MongoDB
    async connect() {
        try {
            this.client = new MongoClient(this.mongoUri);
            await this.client.connect();
            this.db = this.client.db(this.dbName);
            console.log('Connected to MongoDB successfully');
        } catch (error) {
            console.error('MongoDB connection failed:', error.message);
            throw error;
        }
    }

    // Disconnect from MongoDB
    async disconnect() {
        if (this.client) {
            await this.client.close();
            console.log('Disconnected from MongoDB');
        }
    }

    // Preprocess CSV data
    preprocessData(data) {
        return data.map(row => {
            const processedRow = {};
            
            Object.keys(row).forEach(key => {
                // Clean header names
                const cleanKey = key.trim()
                    .replace(/\s+/g, '_')
                    .replace(/[^a-zA-Z0-9_]/g, '')
                    .toLowerCase();
                
                let value = row[key];
                
                // Handle empty values
                if (value === '' || value === null || value === undefined) {
                    processedRow[cleanKey] = null;
                    return;
                }
                
                // Trim whitespace
                if (typeof value === 'string') {
                    value = value.trim();
                }
                
                // Type conversion
                processedRow[cleanKey] = this.convertDataType(value);
            });
            
            // Add metadata
            processedRow._uploaded_at = new Date();
            processedRow._processed = true;
            
            return processedRow;
        });
    }

    // Convert data types
    convertDataType(value) {
        if (typeof value !== 'string') return value;
        
        // Check for boolean
        if (value.toLowerCase() === 'true') return true;
        if (value.toLowerCase() === 'false') return false;
        
        // Check for number
        if (/^\d+$/.test(value)) {
            return parseInt(value, 10);
        }
        
        if (/^\d*\.\d+$/.test(value)) {
            return parseFloat(value);
        }
        
        // Check for date (basic ISO format)
        if (/^\d{4}-\d{2}-\d{2}/.test(value)) {
            const date = new Date(value);
            if (!isNaN(date.getTime())) {
                return date;
            }
        }
        
        return value;
    }

    // Read and parse CSV file
    async readCSV(filePath) {
        return new Promise((resolve, reject) => {
            const results = [];
            const errors = [];
            
            if (!fs.existsSync(filePath)) {
                reject(new Error(`File not found: ${filePath}`));
                return;
            }
            
            fs.createReadStream(filePath)
                .pipe(csv({
                    skipEmptyLines: true,
                    skipLinesWithError: true
                }))
                .on('data', (data) => {
                    results.push(data);
                })
                .on('error', (error) => {
                    errors.push(error);
                })
                .on('end', () => {
                    if (errors.length > 0) {
                        console.warn(`⚠️  ${errors.length} parsing errors occurred`);
                    }
                    console.log(`CSV parsed successfully. ${results.length} rows found`);
                    resolve(results);
                });
        });
    }

    // Upload data to MongoDB
    async uploadToMongo(data, collectionName, options = {}) {
        try {
            const collection = this.db.collection(collectionName);
            
            // Create index on _uploaded_at for better performance
            await collection.createIndex({ "_uploaded_at": 1 });
            
            let result;
            
            if (options.batchSize && data.length > options.batchSize) {
                // Upload in batches for large datasets
                result = await this.uploadInBatches(collection, data, options.batchSize);
            } else {
                // Single batch upload
                result = await collection.insertMany(data, { 
                    ordered: false,
                    ...options 
                });
            }
            
            console.log(`Successfully uploaded ${result.insertedCount || data.length} documents to collection: ${collectionName}`);
            return result;
            
        } catch (error) {
            console.error('Upload failed:', error.message);
            throw error;
        }
    }

    // Upload data in batches
    async uploadInBatches(collection, data, batchSize) {
        const totalBatches = Math.ceil(data.length / batchSize);
        let totalInserted = 0;
        
        console.log(`📦 Uploading ${data.length} records in ${totalBatches} batches of ${batchSize}`);
        
        for (let i = 0; i < totalBatches; i++) {
            const start = i * batchSize;
            const end = Math.min(start + batchSize, data.length);
            const batch = data.slice(start, end);
            
            try {
                const result = await collection.insertMany(batch, { ordered: false });
                totalInserted += result.insertedCount;
                console.log(`Batch ${i + 1}/${totalBatches} completed (${result.insertedCount} records)`);
            } catch (error) {
                console.error(`Batch ${i + 1} failed:`, error.message);
                // Continue with next batch
            }
        }
        
        return { insertedCount: totalInserted };
    }

    // NEW: Get comprehensive collection description
    async getCollectionDescription(collectionName, options = {}) {
        const { sampleSize = 5, includeIndexes = true, includeSchema = true } = options;
        
        try {
            await this.connect();
            const collection = this.db.collection(collectionName);
            
            // Check if collection exists
            const collections = await this.db.listCollections().toArray();
            const collectionExists = collections.some(col => col.name === collectionName);
            
            if (!collectionExists) {
                return { 
                    error: `Collection '${collectionName}' not found in database '${this.dbName}'`,
                    availableCollections: collections.map(col => col.name)
                };
            }
            
            const description = {
                database_name: this.dbName,
                collection_name: collectionName,
                document_count: 0,
                indexes: [],
                sample_documents: [],
                schema_analysis: {},
                field_frequency_analysis: {},
                collection_stats: {},
                creation_info: {}
            };
            
            // Get document count
            description.document_count = await collection.countDocuments();
            
            // Get collection stats
            try {
                const stats = await this.db.command({ collStats: collectionName });
                description.collection_stats = {
                    storage_size: stats.storageSize || 0,
                    total_index_size: stats.totalIndexSize || 0,
                    average_object_size: stats.avgObjSize || 0,
                    size_mb: ((stats.storageSize || 0) / 1024 / 1024).toFixed(2),
                    index_size_mb: ((stats.totalIndexSize || 0) / 1024 / 1024).toFixed(2)
                };
            } catch (error) {
                console.warn('Could not get detailed collection stats');
                description.collection_stats = { error: 'Stats unavailable' };
            }
            
            // Get indexes
            if (includeIndexes) {
                const indexes = await collection.listIndexes().toArray();
                description.indexes = indexes.map(index => ({
                    name: index.name,
                    keys: index.key,
                    unique: index.unique || false,
                    sparse: index.sparse || false,
                    background: index.background || false,
                    partial_filter: index.partialFilterExpression || null
                }));
            }
            
            // Get sample documents and analyze schema
            if (description.document_count > 0) {
                const sampleDocs = await collection.find({}).limit(sampleSize).toArray();
                
                // Clean sample documents (remove _id for display)
                description.sample_documents = sampleDocs.map(doc => {
                    const { _id, ...cleanDoc } = doc;
                    return cleanDoc;
                });
                
                // Analyze schema if requested
                if (includeSchema) {
                    description.schema_analysis = this.analyzeDocumentSchema(sampleDocs[0]);
                    description.field_frequency_analysis = this.analyzeFieldFrequency(sampleDocs);
                }
                
                // Get creation info from first document if it has _uploaded_at
                if (sampleDocs[0]._uploaded_at) {
                    const oldestDoc = await collection.findOne({}, { sort: { _uploaded_at: 1 } });
                    const newestDoc = await collection.findOne({}, { sort: { _uploaded_at: -1 } });
                    
                    description.creation_info = {
                        first_upload: oldestDoc._uploaded_at,
                        last_upload: newestDoc._uploaded_at,
                        has_upload_metadata: true
                    };
                }
            }
            
            return description;
            
        } catch (error) {
            console.error('Failed to get collection description:', error.message);
            return { error: `Error retrieving collection description: ${error.message}` };
        } finally {
            await this.disconnect();
        }
    }

    // NEW: Analyze document schema
    analyzeDocumentSchema(document) {
        if (!document) return {};
        
        const schema = {};
        
        Object.keys(document).forEach(key => {
            if (key === '_id') return;
            
            const value = document[key];
            const valueType = this.getValueType(value);
            
            if (typeof value === 'object' && value !== null && !Array.isArray(value) && !(value instanceof Date)) {
                schema[key] = {
                    type: 'object',
                    nested_fields: Object.keys(value),
                    sample_structure: this.getObjectStructure(value)
                };
            } else if (Array.isArray(value)) {
                schema[key] = {
                    type: 'array',
                    length: value.length,
                    item_type: value.length > 0 ? this.getValueType(value[0]) : 'unknown',
                    sample_items: value.slice(0, 3) // First 3 items as sample
                };
            } else {
                schema[key] = {
                    type: valueType,
                    sample_value: this.getSampleValue(value),
                    is_nullable: value === null
                };
            }
        });
        
        return schema;
    }

    // NEW: Analyze field frequency across documents
    analyzeFieldFrequency(documents) {
        const fieldCount = {};
        const totalDocs = documents.length;
        
        documents.forEach(doc => {
            Object.keys(doc).forEach(key => {
                if (key === '_id') return;
                fieldCount[key] = (fieldCount[key] || 0) + 1;
            });
        });
        
        const fieldAnalysis = {};
        Object.keys(fieldCount).forEach(field => {
            const count = fieldCount[field];
            fieldAnalysis[field] = {
                frequency: count,
                percentage: Math.round((count / totalDocs) * 100),
                is_common: count > (totalDocs * 0.8), // Appears in >80% of docs
                consistency: count === totalDocs ? 'always_present' : 
                           count > (totalDocs * 0.8) ? 'mostly_present' : 'sometimes_present'
            };
        });
        
        return fieldAnalysis;
    }

    // Helper: Get value type
    getValueType(value) {
        if (value === null) return 'null';
        if (Array.isArray(value)) return 'array';
        if (value instanceof Date) return 'date';
        if (typeof value === 'object') return 'object';
        return typeof value;
    }

    // Helper: Get object structure
    getObjectStructure(obj) {
        const structure = {};
        Object.keys(obj).forEach(key => {
            structure[key] = this.getValueType(obj[key]);
        });
        return structure;
    }

    // Helper: Get sample value for display
    getSampleValue(value) {
        if (value === null || value === undefined) return null;
        if (typeof value === 'string' && value.length > 50) {
            return value.substring(0, 50) + '...';
        }
        if (value instanceof Date) {
            return value.toISOString();
        }
        return value;
    }

    // NEW: Pretty print collection description
    async printCollectionDescription(collectionName, options = {}) {
        console.log(`\nGetting description for collection: ${collectionName}`);
        console.log('═'.repeat(60));
        
        const description = await this.getCollectionDescription(collectionName, options);
        
        if (description.error) {
            console.error(`    ${description.error}`);
            if (description.availableCollections) {
                console.log(`Available collections: ${description.availableCollections.join(', ')}`);
            }
            return description;
        }
        
        // Basic Info
        console.log(`\nBASIC INFORMATION`);
        console.log(`   Database: ${description.database_name}`);
        console.log(`   Collection: ${description.collection_name}`);
        console.log(`   Document Count: ${description.document_count.toLocaleString()}`);
        
        // Storage Stats
//        if (description.collection_stats && !description.collection_stats.error) {
  //          console.log(`\n STORAGE STATISTICS`);
    //        console.log(`   Storage Size: ${description.collection_stats.size_mb} MB`);
      //      console.log(`   Index Size: ${description.collection_stats.index_size_mb} MB`);
        //    console.log(`   Avg Document Size: ${description.collection_stats.average_object_size} bytes`);
        //}
        
        // Creation Info
//      if (description.creation_info && description.creation_info.has_upload_metadata) {
  //          console.log(`\nUPLOAD INFORMATION`);
    //        console.log(`   First Upload: ${description.creation_info.first_upload}`);
      //      console.log(`   Last Upload: ${description.creation_info.last_upload}`);
       // }
        
        // Indexes
        if (description.indexes && description.indexes.length > 0) {
            console.log(`\n🔗 INDEXES (${description.indexes.length})`);
            description.indexes.forEach(index => {
                console.log(`   • ${index.name}: ${JSON.stringify(index.keys)} ${index.unique ? '(unique)' : ''}`);
            });
        }
        
        // Schema Analysis
        if (description.schema_analysis && Object.keys(description.schema_analysis).length > 0) {
            console.log(`\n SCHEMA ANALYSIS`);
            Object.keys(description.schema_analysis).forEach(field => {
                const fieldInfo = description.schema_analysis[field];
                console.log(`   • ${field}: ${fieldInfo.type} ${fieldInfo.sample_value ? `(e.g., ${fieldInfo.sample_value})` : ''}`);
            });
        }
        
        // Field Frequency
//        if (description.field_frequency_analysis && Object.keys(description.field_frequency_analysis).length > 0) {
  //          console.log(`\nFIELD FREQUENCY ANALYSIS`);
    //        Object.keys(description.field_frequency_analysis).forEach(field => {
      //          const freq = description.field_frequency_analysis[field];
        //        console.log(`   • ${field}: ${freq.percentage}% (${freq.consistency})`);
          //  });
        //}
        
        // Sample Documents
  //      if (description.sample_documents && description.sample_documents.length > 0) {
    //        console.log(`\n📄 SAMPLE DOCUMENTS (${description.sample_documents.length})`);
      //      description.sample_documents.forEach((doc, index) => {
        //        console.log(`   Sample ${index + 1}:`, JSON.stringify(doc, null, 2).substring(0, 200) + '...');
          //  });
        //}
        
        console.log('\n' + '═'.repeat(60));
        return description;
    }

    // Main upload function
    async uploadCSV(filePath, collectionName, options = {}) {
        const startTime = Date.now();
        
        try {
            console.log(`Starting CSV upload process...`);
            console.log(` File: ${filePath}`);
            console.log(`  Collection: ${collectionName}`);
            
            // Connect to MongoDB
            await this.connect();
            
            // Read CSV
            console.log(`Reading CSV file...`);
            const rawData = await this.readCSV(filePath);
            
            if (rawData.length === 0) {
                throw new Error('CSV file is empty or contains no valid data');
            }
            
            // Preprocess data
            console.log(`Preprocessing data...`);
            const processedData = this.preprocessData(rawData);
            
            // Upload to MongoDB
            console.log(` Uploading to MongoDB...`);
            const result = await this.uploadToMongo(processedData, collectionName, {
                batchSize: options.batchSize || 1000,
                ...options
            });
            
            const endTime = Date.now();
            const duration = ((endTime - startTime) / 1000).toFixed(2);
            
            console.log(`\n Upload completed successfully!`);
            console.log(` Statistics:`);
            console.log(`   • Total records: ${rawData.length}`);
            console.log(`   • Successfully uploaded: ${result.insertedCount || processedData.length}`);
            console.log(`   • Time taken: ${duration} seconds`);
            console.log(`   • Collection: ${collectionName}`);
            
            return result;
            
        } catch (error) {
            console.error(` Upload failed:`, error.message);
            throw error;
        } finally {
            await this.disconnect();
        }
    }

    // Utility: Get collection stats (Enhanced version)
    async getCollectionStats(collectionName) {
        try {
            await this.connect();
            const collection = this.db.collection(collectionName);
            
            // Get document count
            const count = await collection.countDocuments();
            
            // Get collection info using admin command
            let stats = {};
            try {
                const collStats = await this.db.command({ collStats: collectionName });
                stats = {
                    storageSize: collStats.storageSize || 0,
                    avgObjSize: collStats.avgObjSize || 0,
                    totalIndexSize: collStats.totalIndexSize || 0
                };
            } catch (adminError) {
                console.warn(' Could not get detailed storage stats');
                stats = {
                    storageSize: 0,
                    avgObjSize: 0,
                    totalIndexSize: 0
                };
            }
            
            // Get sample document to estimate size
            const sampleDoc = await collection.findOne();
            const estimatedDocSize = sampleDoc ? JSON.stringify(sampleDoc).length : 0;
            
            console.log(` Collection Stats for '${collectionName}':`);
            console.log(`   • Document count: ${count.toLocaleString()}`);
            console.log(`   • Storage size: ${(stats.storageSize / 1024 / 1024).toFixed(2)} MB`);
            console.log(`   • Index size: ${(stats.totalIndexSize / 1024 / 1024).toFixed(2)} MB`);
            console.log(`   • Avg document size: ${stats.avgObjSize || estimatedDocSize} bytes`);
            
            return { 
                count, 
                storageSize: stats.storageSize,
                indexSize: stats.totalIndexSize,
                avgDocSize: stats.avgObjSize || estimatedDocSize
            };
            
        } catch (error) {
            console.error(' Failed to get collection stats:', error.message);
            throw error;
        } finally {
            await this.disconnect();
        }
    }
}

// Usage Example and Configuration
async function main() {
    // Configuration
    const config = {
        mongoUri: 'mongodb+srv://user:password@cluster0.bptf95x.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0',
        dbName: 'CSV_Analyzer',
        csvFilePath: 'D://current projects//llm+rag//data//sales_data.csv',
        collectionName: 'CSV_Analyzer2',
        batchSize: 1000
    };
    
    const uploader = new CSVMongoUploader(config.mongoUri, config.dbName);
    
    try {
        // Upload CSV to MongoDB
        await uploader.uploadCSV(config.csvFilePath, config.collectionName, {
            batchSize: config.batchSize
        });
        
        // Get detailed collection description
        console.log('\n' + '='.repeat(80));
        console.log('GETTING DETAILED COLLECTION DESCRIPTION');
        console.log('='.repeat(80));
        
        await uploader.printCollectionDescription(config.collectionName, {
            sampleSize: 3,
            includeIndexes: true,
            includeSchema: true
        });
        
        // You can also get the raw description object
        const description = await uploader.getCollectionDescription(config.collectionName);
        // console.log('Raw description object:', JSON.stringify(description, null, 2));
        
    } catch (error) {
        console.error('Process failed:', error.message);
        process.exit(1);
    }
}

// Export for use as module
module.exports = CSVMongoUploader;

// Run if called directly
if (require.main === module) {
    main();
}
