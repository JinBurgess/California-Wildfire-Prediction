from common_imports import *

class BalancingMethods:
    """
    A class for rebalancing datasets using oversampling or undersampling techniques.
    """

    @staticmethod
    def infrequent_over_sample(locations, df, by="name"):
        """
        Balance the dataset by applying infrequent oversampling or undersampling per city.

        Args:
            df (pd.DataFrame): Input dataset containing the data.
            cities (list): List of unique city names.

        Returns:
            pd.DataFrame: Balanced dataset.
        """
        balanced_df = pd.DataFrame()

        for loc in locations:
            loc_df = df[df[by] == loc] # Filter data for each city or county

            occurred = loc_df[loc_df['FireOccurred'] == 1]
            non_occurred = loc_df[loc_df['FireOccurred'] == 0]

            occurred.sort_values(by='datetime')
            non_occurred.sort_values(by='datetime')

            if len(non_occurred) == 0 or len(occurred) == 0:
                adj_df = loc_df
            elif len(non_occurred) > len(occurred):
                ratio = len(non_occurred) / len(occurred)

                if ratio >= 2:
                    step = int(np.floor(ratio))
                    adj_non_occurred = non_occurred.iloc[::step]
                else:
                    adj_non_occurred = non_occurred

                adj_df = pd.concat([occurred, adj_non_occurred], ignore_index=True)

            elif len(occurred) > len(non_occurred):
                ratio = len(occurred) / len(non_occurred)

                if ratio >= 2:
                    step = int(np.floor(ratio))
                    adj_occurred = occurred.iloc[::step]
                else:
                    adj_occurred = occurred

                adj_df = pd.concat([non_occurred, adj_occurred], ignore_index=True)
            else:
                adj_df = loc_df

            balanced_df = pd.concat([balanced_df, adj_df], ignore_index=True)

        return balanced_df

    @staticmethod
    def sliding_over_sample(locations, df, by = 'name'):
        """
        Balance the dataset by applying sliding window oversampling or undersampling per city.

        Args:
            df (pd.DataFrame): Input dataset containing the data.
            cities (list): List of unique city names.

        Returns:
            pd.DataFrame: Balanced dataset.
        """
        balanced_df = pd.DataFrame()

        for loc in locations:
            loc_df = df[df[by] == loc]

            occurred = loc_df[loc_df['FireOccurred'] == 1]
            non_occurred = loc_df[loc_df['FireOccurred'] == 0]

            occurred.sort_values(by='datetime')
            non_occurred.sort_values(by='datetime')

            if len(non_occurred) == 0 or len(occurred) == 0:
                balanced_df = pd.concat([balanced_df, loc_df], ignore_index=True)
                continue

            if len(non_occurred) > len(occurred):
                ratio = len(non_occurred) / len(occurred)
                step = int(np.floor(ratio)) if ratio >= 2 else 1
                aggregated_non_occurred = []

                for i in range(0, len(non_occurred), step):
                    window = non_occurred.iloc[i:i + step]
                    numeric_agg = window.mean(numeric_only=True)
                    name_agg = window['name'].iloc[0]
                    datetime_agg = window['datetime'].iloc[0]
                    aggregated_non_occurred.append(
                        pd.concat([numeric_agg, pd.Series({'name': name_agg, 'datetime': datetime_agg})])
                    )

                adj_non_occurred = pd.DataFrame(aggregated_non_occurred)
                adj_df = pd.concat([occurred, adj_non_occurred], ignore_index=True)

            elif len(occurred) > len(non_occurred):
                ratio = len(occurred) / len(non_occurred)
                step = int(np.floor(ratio)) if ratio >= 2 else 1
                aggregated_occurred = []

                for i in range(0, len(occurred), step):
                    window = occurred.iloc[i:i + step]
                    numeric_agg = window.mean(numeric_only=True)
                    name_agg = window['name'].iloc[0]
                    datetime_agg = window['datetime'].iloc[0]
                    aggregated_occurred.append(
                        pd.concat([numeric_agg, pd.Series({'name': name_agg, 'datetime': datetime_agg})])
                    )

                adj_occurred = pd.DataFrame(aggregated_occurred)
                adj_df = pd.concat([non_occurred, adj_occurred], ignore_index=True)

            else:
                adj_df = loc_df

            balanced_df = pd.concat([balanced_df, adj_df], ignore_index=True)

        balanced_df['FireOccurred'] = balanced_df['FireOccurred'].astype(int)
        return balanced_df
