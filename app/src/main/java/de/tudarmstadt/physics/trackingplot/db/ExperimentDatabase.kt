package de.tudarmstadt.physics.trackingplot.db

import android.content.Context
import android.database.sqlite.SQLiteDatabase
import android.database.sqlite.SQLiteException
import android.database.sqlite.SQLiteOpenHelper
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

class ExperimentDatabase private constructor(context: Context)
    : SQLiteOpenHelper(context, DATABASE_NAME, null, DATABASE_VERSION) {

    companion object {
        private const val DATABASE_NAME = "experiments.db"
        private const val DATABASE_VERSION = 1

        @Volatile
        private var instance: ExperimentDatabase? = null

        fun getInstance(context: Context): ExperimentDatabase {
            return instance ?: synchronized(this) {
                instance ?: ExperimentDatabase(context.applicationContext).also { instance = it }
            }
        }
    }

    override fun onCreate(db: SQLiteDatabase) {

        db.execSQL("CREATE TABLE Experiment (" +
                "ExperimentId INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL" +
                ", Description TEXT NOT NULL" +
                //todo bounding box config
                //todo ruler config
                ")")

        db.execSQL("CREATE TABLE Ruler (" +
                "ExperimentId INTEGER PRIMARY KEY NOT NULL" +
                ", P1X REAL NOT NULL" +
                ", P1Y REAL NOT NULL" +
                ", P2X REAL NOT NULL" +
                ", P2Y REAL NOT NULL" +
                ", Distance REAL NOT NULL" +
                ", Unit TEXT NOT NULL" +
                //todo tracker config
                ", FOREIGN KEY (ExperimentId) REFERENCES Experiment(ExperimentId) ON DELETE CASCADE" +
                ")")

        db.execSQL("CREATE TABLE Roi (" +
                "ExperimentId INTEGER PRIMARY KEY NOT NULL" +
                ", P1X REAL NOT NULL" +
                ", P1Y REAL NOT NULL" +
                ", P2X REAL NOT NULL" +
                ", P2Y REAL NOT NULL" +
                //todo tracker config
                ", FOREIGN KEY (ExperimentId) REFERENCES Experiment(ExperimentId) ON DELETE CASCADE" +
                ")")

        db.execSQL("CREATE TABLE Tracker (" +
//                "TrackerId INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL" +
                "ExperimentId INTEGER NOT NULL" +
                ", TrackerIndex INTEGER NOT NULL" +
                ", Color INTEGER NOT NULL" +
                ", Tolerance REAL NOT NULL" +
                ", PRIMARY KEY (ExperimentId, TrackerIndex)" +
                ", FOREIGN KEY (ExperimentId) REFERENCES Experiment(ExperimentId) ON DELETE CASCADE" +
                ")")

        db.execSQL("CREATE TABLE Measurement (" +
                "ExperimentId INTEGER NOT NULL" +
                ", TrackerIndex INTEGER NOT NULL" +
                ", Timestamp INTEGER NOT NULL" +
                ", CentroidX REAL NOT NULL" +
                ", CentroidY REAL NOT NULL" +
                ", PRIMARY KEY (ExperimentId, TrackerIndex, Timestamp)" +
                ", FOREIGN KEY (ExperimentId, TrackerIndex) REFERENCES Tracker(ExperimentId, TrackerIndex) ON DELETE CASCADE" +
                ")")
    }

    override fun onUpgrade(db: SQLiteDatabase, oldVersion: Int, newVersion: Int) {
        /* no-op */ // implement when needed
    }

//    override fun onDowngrade(db: SQLiteDatabase, oldVersion: Int, newVersion: Int) {
//        super.onDowngrade(db, oldVersion, newVersion)
//    }

    suspend fun <R> withTransaction(
        readOnly: Boolean,
        block: TransactionScope.() -> R
    ): R {
        val transactionBlock: CoroutineScope.() -> R = txn@{
            val db = if (readOnly) readableDatabase else writableDatabase
            if (readOnly) db.beginTransactionReadOnly()
            else db.beginTransaction()

            try {
                val scope = TransactionScope(db)
                val result = try {
                    scope.block()
                } catch (e: SQLiteException) {
                    throw DatabaseException(e)
                }

                db.setTransactionSuccessful()

                return@txn result
            } finally {
                db.endTransaction()
            }
        }
        return withContext(Dispatchers.IO, transactionBlock)
    }
}