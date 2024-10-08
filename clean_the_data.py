{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "c35c2b76-3ea7-4148-9392-5683dc8a49a4",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "0d2f4e89-4686-4c5d-af43-4ed4b3f10d1c",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "       AAs AA1 AA2 AA3 AA4  # Stop   fitness  active\n",
      "9594  AAAA   A   A   A   A     0.0  0.074455    True\n",
      "9595  AAAC   A   A   A   C     0.0  0.056314    True\n",
      "9596  AAAD   A   A   A   D     0.0  0.014342   False\n",
      "9597  AAAE   A   A   A   E     0.0  0.012914   False\n",
      "9598  AAAF   A   A   A   F     0.0  0.005161   False\n"
     ]
    }
   ],
   "source": [
    "# Load the CSV file into a DataFrame\n",
    "df = pd.read_csv('data/four-site_simplified_AA_data.csv')\n",
    "\n",
    "# this will get rid of all data with more than 1 spot seqence\n",
    "filtered_df = df.loc[df['# Stop'] <= 0.0]\n",
    "\n",
    "# Print the filtered DataFrame to verify\n",
    "print(filtered_df.head())\n",
    "\n",
    "df.to_csv('data/four-site_clean_simplified_AA_data.csv', index=False)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "fb553083-88e4-4654-889c-9f515ed8ca7b",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "9782b324-0225-43dd-af8b-165827e82df2",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.6"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
