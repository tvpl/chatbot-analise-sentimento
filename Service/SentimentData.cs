using Microsoft.ML.Data;

namespace AnaliseSentimentoChatBot.Service
{
    public class SentimentData
    {
        [LoadColumn(0)]
        public string SentimentText { get; set; }

        [LoadColumn(1), ColumnName("Label")]
        public bool Sentiment { get; set; }
    }
}
