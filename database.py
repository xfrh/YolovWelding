import pymongo
class Database(object):
    URI="localhost:27017"
    DATABASE=None

    @staticmethod
    def initialize():
        client = pymongo.MongoClient(Database.URI)
        Database.DATABASE = client["RD_DB"]
    @staticmethod
    def insert(collection,data):
        return Database.DATABASE[collection].insert_one(data)
    @staticmethod
    def update(collection,query,newvalue):
       return  Database.DATABASE[collection].update_one(query,newvalue)
    @staticmethod
    def find(collection,query):
        return Database.DATABASE[collection].find(query)

    @staticmethod
    def findone(collection, query):
        return Database.DATABASE[collection].find_one(query)
    @staticmethod
    def deleteone(collection,query):
       return  Database.DATABASE[collection].delete_one(query)