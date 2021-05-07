using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.Data.SqlClient;
using System.IO;
using System.Threading.Tasks;
using static Microsoft.ML.DataOperationsCatalog;

namespace AnaliseSentimentoChatBot.Service
{
    public class SentimentAnalysis
    {
        static readonly string _dataPath = Path.Combine(Environment.CurrentDirectory, "Data", "data.txt");

        private static string _connectionString = @"Data Source=SQL5103.site4now.net;Initial Catalog=DB_A6A282_tcc;User Id=DB_A6A282_tcc_admin;Password=tvpl1991";

        private static MLContext Context { get; set; }

        private static ITransformer Model { get; set; }

        private static PredictionEngine<SentimentData, SentimentPrediction> Engine { get; set; }

        static SentimentAnalysis()
        {
            Context = new MLContext();

            //Carrega e treinar nosso modelo
            TrainTestData splitDataView = LoadData();

            Model = BuildAndTrainModel(splitDataView.TrainSet);
        }

        private static TrainTestData LoadData()
        {
            DatabaseLoader loader = Context.Data.CreateDatabaseLoader<SentimentData>();

            string sqlCommand = "SELECT SentimentText, Sentiment FROM Dataset WHERE InTraining = 1";

            DatabaseSource dbSource = new DatabaseSource(SqlClientFactory.Instance, _connectionString, sqlCommand);

            //IDataView dataView = Context.Data.LoadFromTextFile<SentimentData>(_dataPath, hasHeader: false);
            IDataView dataView = loader.Load(dbSource);

            TrainTestData splitDataView = Context.Data.TrainTestSplit(dataView, testFraction: 0.65);
            return splitDataView;
        }

        private static ITransformer BuildAndTrainModel(IDataView trainSet)
        {
            var estimator = Context.Transforms.Text.FeaturizeText(outputColumnName: "Features", inputColumnName: nameof(SentimentData.SentimentText))
                .Append(Context.BinaryClassification.Trainers.FastTree(labelColumnName: "Label", featureColumnName: "Features"));

            //Cria e treina o modelo
            var model = estimator.Fit(trainSet);
            return model;
        }

        private static PredictionEngine<SentimentData, SentimentPrediction> GetEngine()
        {
            if (Engine != null)
                return Engine;

            Engine = Context.Model.CreatePredictionEngine<SentimentData, SentimentPrediction>(Model);

            return Engine;
        }

        public static void Retraining()
        {
            Engine = Context.Model.CreatePredictionEngine<SentimentData, SentimentPrediction>(Model);
        }

        public static SentimentPrediction GetPrediction(string text)
        {
            var engine = GetEngine();

            var prediction = engine.Predict(new SentimentData()
            {
                SentimentText = text
            });

            return prediction;
        }

        public static async Task TrainingAsync(string text, int prediction)
        {
            if (!string.IsNullOrEmpty(text))
            {
                try
                {
                    string sql = $"INSERT INTO Dataset(SentimentText, Sentiment, DateInclusion, WeightFeeling, InTraining) VALUES('{text}', {prediction}, GETDATE(), 1, 1)";
                    await ExecuteQueryAsync(sql);
                }
                catch (Exception ex)
                {
                    Console.WriteLine(ex.Message);
                }
            }
        }

        public static async Task RegisterPredictionAsync(string text, int prediction, bool resultAlgorithmPrediction)
        {
            if (!string.IsNullOrEmpty(text) && prediction < 2)
            {
                try
                {
                    string sqlDataSet = $"INSERT INTO Dataset(SentimentText, Sentiment, DateInclusion, WeightFeeling, InTraining) VALUES('{text}', {prediction}, GETDATE(), 1, 1)";
                    string sqlStatisticalResult = $"INSERT INTO StatisticalResults(CPF, DateOccurrence, [Result]) VALUES('-', GETDATE(), {(resultAlgorithmPrediction ? 1 : 0)})";

                    await ExecuteQueryAsync(sqlDataSet);
                    await ExecuteQueryAsync(sqlStatisticalResult);
                }
                catch (Exception ex)
                {
                    Console.WriteLine(ex.Message);
                }
            }
        }

        private static async Task ExecuteQueryAsync(string sql)
        {
            SqlConnection conn = new SqlConnection(_connectionString);
            SqlCommand comando = new SqlCommand(sql, conn);
            conn.Open();
            await comando.ExecuteNonQueryAsync();
            conn.Close();
        }
    }
}
