from moto import mock_s3
import boto3
import json


class s3_bucket:
    def __init__(self, region, bucket_name):
        self.region = region
        self.bucket_name = bucket_name
        self.s3 = boto3.resource("s3", region_name=self.region)

        self.verbs_fld = "verbs/"
        self.users_fld = "users/"

    def create(self):
        self.s3.create_bucket(Bucket=self.bucket_name)

    def creation_date(self):
        return self.s3.Bucket(self.bucket_name).creation_date

    def obj_exists(self, path) -> bool:
        for _ in self.s3.Bucket(self.bucket_name).objects.filter(Prefix=path):
            return True
        return False

    def dict_from_s3(self, filename) -> dict:
        json_file = self.json_file_from_s3(filename)
        return json.loads(json_file)  # dict

    def json_file_from_s3(self, filename) -> str:
        obj = self.s3.Object(self.bucket_name, filename)
        return obj.get()["Body"].read().decode("utf-8")

    def dict_to_s3(self, filename, pydict) -> None:
        json_file = json.dumps(pydict)
        self.json_file_to_s3(json_file, filename)

    def json_file_to_s3(self, json_file, filename) -> None:
        obj = self.s3.Object(self.bucket_name, filename)
        # json_bytes = bytes(json.dumps(json_file).encode("UTF-8"))
        obj.put(Body=json_file)


@mock_s3
def main():

    s3_b = s3_bucket("us-east-1", "streamlit-conjuga")

    # if bucket does not exist, create
    if s3_b.creation_date() is None:
        s3_b.create()
