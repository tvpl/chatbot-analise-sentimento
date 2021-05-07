using Microsoft.ML.Data;

namespace AnaliseSentimentoChatBot.Service
{
    public class SentimentPrediction
    {
        [ColumnName("PredictedLabel")]
        public bool Prediction { get; set; }

        public float Probability { get; set; }

        public float Score { get; set; }

        public float Percentage => Probability * 100;
    }
}
