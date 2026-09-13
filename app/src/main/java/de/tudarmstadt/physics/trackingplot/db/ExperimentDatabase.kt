package de.tudarmstadt.physics.trackingplot.db

import android.content.Context
import android.database.sqlite.SQLiteDatabase
import android.database.sqlite.SQLiteException
import android.database.sqlite.SQLiteOpenHelper
import android.os.Build
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

class ExperimentDatabase private constructor(context: Context) :
    SQLiteOpenHelper(context, DATABASE_NAME, null, DATABASE_VERSION) {

    companion object {
        private const val DATABASE_NAME = "experiments.db"

        /**
         * 1: initial schema.
         * 2: Experiment gains CreatedAt, SamplingRate and ZoomRatio.
         */
        private const val DATABASE_VERSION = 2

        @Volatile
        private var instance: ExperimentDatabase? = null

        fun getInstance(context: Context): ExperimentDatabase {
            return instance ?: synchronized(this) {
                instance ?: ExperimentDatabase(context.applicationContext).also { instance = it }
            }
        }
    }

    init {
        // Readers do not block the measurement writer and vice versa.
        setWriteAheadLoggingEnabled(true)
    }

    override fun onConfigure(db: SQLiteDatabase) {
        db.setForeignKeyConstraintsEnabled(true)
    }

    override fun onCreate(db: SQLiteDatabase) {
        db.execSQL(
            "CREATE TABLE Experiment (" +
                "ExperimentId INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL" +
                ", Description TEXT NOT NULL" +
                ", CreatedAt INTEGER NOT NULL DEFAULT 0" +
                ", SamplingRate INTEGER NOT NULL DEFAULT 0" +
                ", ZoomRatio REAL NOT NULL DEFAULT 1.0" +
                ")"
        )

        db.execSQL(
            "CREATE TABLE Ruler (" +
                "ExperimentId INTEGER PRIMARY KEY NOT NULL" +
                ", P1X REAL NOT NULL" +
                ", P1Y REAL NOT NULL" +
                ", P2X REAL NOT NULL" +
                ", P2Y REAL NOT NULL" +
                ", Distance REAL NOT NULL" +
                ", Unit TEXT NOT NULL" +
                ", FOREIGN KEY (ExperimentId) REFERENCES Experiment(ExperimentId) ON DELETE CASCADE" +
                ")"
        )

        db.execSQL(
            "CREATE TABLE Roi (" +
                "ExperimentId INTEGER PRIMARY KEY NOT NULL" +
                ", P1X REAL NOT NULL" +
                ", P1Y REAL NOT NULL" +
                ", P2X REAL NOT NULL" +
                ", P2Y REAL NOT NULL" +
                ", FOREIGN KEY (ExperimentId) REFERENCES Experiment(ExperimentId) ON DELETE CASCADE" +
                ")"
        )

        db.execSQL(
            "CREATE TABLE Tracker (" +
                "ExperimentId INTEGER NOT NULL" +
                ", TrackerIndex INTEGER NOT NULL" +
                ", Color INTEGER NOT NULL" +
                ", Tolerance REAL NOT NULL" +
                ", PRIMARY KEY (ExperimentId, TrackerIndex)" +
                ", FOREIGN KEY (ExperimentId) REFERENCES Experiment(ExperimentId) ON DELETE CASCADE" +
                ")"
        )

        db.execSQL(
            "CREATE TABLE Measurement (" +
                "ExperimentId INTEGER NOT NULL" +
                ", TrackerIndex INTEGER NOT NULL" +
                ", Timestamp INTEGER NOT NULL" +
                ", CentroidX REAL NOT NULL" +
                ", CentroidY REAL NOT NULL" +
                ", PRIMARY KEY (ExperimentId, TrackerIndex, Timestamp)" +
                ", FOREIGN KEY (ExperimentId, TrackerIndex) REFERENCES Tracker(ExperimentId, TrackerIndex) ON DELETE CASCADE" +
                ")"
        )
    }

    override fun onUpgrade(db: SQLiteDatabase, oldVersion: Int, newVersion: Int) {
        if (oldVersion < 2) {
            db.execSQL("ALTER TABLE Experiment ADD COLUMN CreatedAt INTEGER NOT NULL DEFAULT 0")
            db.execSQL("ALTER TABLE Experiment ADD COLUMN SamplingRate INTEGER NOT NULL DEFAULT 0")
            db.execSQL("ALTER TABLE Experiment ADD COLUMN ZoomRatio REAL NOT NULL DEFAULT 1.0")
        }
    }

    override fun onDowngrade(db: SQLiteDatabase, oldVersion: Int, newVersion: Int) {
        // Newer schemas are supersets of older ones and all queries name their columns,
        // so an older build can keep using the database as it is.
    }

    /** Runs [block] inside a read-only transaction on the IO dispatcher. */
    suspend fun <R> read(block: TransactionScope.() -> R): R = transaction(readOnly = true, block)

    /** Runs [block] inside a write transaction on the IO dispatcher. */
    suspend fun <R> write(block: TransactionScope.() -> R): R = transaction(readOnly = false, block)

    private suspend fun <R> transaction(readOnly: Boolean, block: TransactionScope.() -> R): R =
        withContext(Dispatchers.IO) {
            val db = try {
                if (readOnly) readableDatabase else writableDatabase
            } catch (e: SQLiteException) {
                throw DatabaseException("Could not open the database", e)
            }
            try {
                when {
                    readOnly && Build.VERSION.SDK_INT >= Build.VERSION_CODES.VANILLA_ICE_CREAM ->
                        db.beginTransactionReadOnly()
                    readOnly -> db.beginTransactionNonExclusive()
                    else -> db.beginTransactionNonExclusive()
                }
                try {
                    val result = TransactionScope(db).block()
                    db.setTransactionSuccessful()
                    result
                } finally {
                    db.endTransaction()
                }
            } catch (e: SQLiteException) {
                // Covers query errors as well as failures at commit time (disk full, I/O errors).
                throw DatabaseException(e.message ?: "Database error", e)
            }
        }
}
