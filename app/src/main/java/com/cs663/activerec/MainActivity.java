package com.cs663.activerec;

import android.content.Intent;
import android.os.Bundle;
import android.view.View;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.Spinner;

import androidx.appcompat.app.AppCompatActivity;

/**
 * Spinner of options where "ActiveRec" opens the LSTM activity recognition screen.
 * To add ActiveRec to the group's existing app, add "ActiveRec" to its spinner and
 * copy the "ActiveRec" branch below.
 */
public class MainActivity extends AppCompatActivity {

    private static final String[] OPTIONS = {"Select an option", "ActiveRec"};

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        Spinner spinner = findViewById(R.id.modeSpinner);
        spinner.setAdapter(new ArrayAdapter<>(this,
                android.R.layout.simple_spinner_dropdown_item, OPTIONS));

        spinner.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> parent, View view, int position, long id) {
                String selected = OPTIONS[position];
                if (selected.equals("ActiveRec")) {
                    startActivity(new Intent(MainActivity.this, ActiveRecActivity.class));
                    spinner.setSelection(0);   // so choosing ActiveRec again re-opens it
                }
            }

            @Override
            public void onNothingSelected(AdapterView<?> parent) {
            }
        });
    }
}
